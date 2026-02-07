#!/usr/bin/env python3
"""Android Super Partition Toolkit.

Inspect, extract, modify, and repack logical partitions (vendor, system, etc.)
inside Android GPT disk images with dynamic partitions (super).

Requires: erofs-utils (mkfs.erofs, fsck.erofs) for EROFS partition operations.
"""

import argparse
import glob as globmod
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# GPT Parser
# ---------------------------------------------------------------------------

GPT_HEADER_MAGIC = b'EFI PART'
SECTOR = 512


class GPTEntry:
    def __init__(self, name, start_lba, end_lba, type_guid, unique_guid, attributes):
        self.name = name
        self.start_lba = start_lba
        self.end_lba = end_lba
        self.size_sectors = end_lba - start_lba + 1
        self.type_guid = type_guid
        self.unique_guid = unique_guid
        self.attributes = attributes


class GPTParser:
    def __init__(self, image_path):
        self.image_path = image_path
        self.entries = []
        self._parse()

    def _parse(self):
        with open(self.image_path, 'rb') as f:
            # GPT header at LBA 1
            f.seek(SECTOR)
            hdr = f.read(SECTOR)
            if hdr[:8] != GPT_HEADER_MAGIC:
                raise ValueError('Not a GPT disk image')

            entry_start_lba = struct.unpack_from('<Q', hdr, 72)[0]
            entry_count = struct.unpack_from('<I', hdr, 80)[0]
            entry_size = struct.unpack_from('<I', hdr, 84)[0]

            f.seek(entry_start_lba * SECTOR)
            for _ in range(entry_count):
                raw = f.read(entry_size)
                type_guid = raw[0:16]
                unique_guid = raw[16:32]
                start_lba = struct.unpack_from('<Q', raw, 32)[0]
                end_lba = struct.unpack_from('<Q', raw, 40)[0]
                attributes = struct.unpack_from('<Q', raw, 48)[0]
                name = raw[56:128].decode('utf-16-le').rstrip('\x00')

                if start_lba == 0 and end_lba == 0:
                    continue
                self.entries.append(GPTEntry(name, start_lba, end_lba,
                                             type_guid, unique_guid, attributes))

    def find_partition(self, name):
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def list_partitions(self):
        return list(self.entries)


# ---------------------------------------------------------------------------
# LP Metadata (Android Dynamic Partitions)
# ---------------------------------------------------------------------------

LP_METADATA_GEOMETRY_MAGIC = 0x616C4467
LP_METADATA_HEADER_MAGIC = 0x414C5030


class LPPartition:
    def __init__(self, name, attrs, first_extent, num_extents, group_index):
        self.name = name
        self.attrs = attrs
        self.first_extent = first_extent
        self.num_extents = num_extents
        self.group_index = group_index

    @property
    def readonly(self):
        return bool(self.attrs & 1)

    def total_sectors(self, extents):
        total = 0
        for i in range(self.num_extents):
            total += extents[self.first_extent + i].num_sectors
        return total


class LPExtent:
    TARGET_LINEAR = 0
    TARGET_ZERO = 1

    def __init__(self, num_sectors, target_type, target_data, target_source):
        self.num_sectors = num_sectors
        self.target_type = target_type
        self.target_data = target_data  # physical sector offset for LINEAR
        self.target_source = target_source

    @property
    def size_bytes(self):
        return self.num_sectors * SECTOR


class LPGroup:
    def __init__(self, name, flags, max_size):
        self.name = name
        self.flags = flags
        self.max_size = max_size


class LPBlockDevice:
    def __init__(self, first_logical_sector, alignment, alignment_offset, size, name, flags):
        self.first_logical_sector = first_logical_sector
        self.alignment = alignment
        self.alignment_offset = alignment_offset
        self.size = size
        self.name = name
        self.flags = flags


class LPMetadata:
    def __init__(self):
        self.partitions = []
        self.extents = []
        self.groups = []
        self.block_devices = []
        self.header_size = 256
        self.major = 10
        self.minor = 2
        self._tables_size = 0
        self._raw_header = None

    @classmethod
    def parse(cls, data):
        meta = cls()
        meta._raw = bytearray(data)

        magic = struct.unpack_from('<I', data, 0)[0]
        if magic != LP_METADATA_HEADER_MAGIC:
            raise ValueError(f'Bad LP metadata magic: 0x{magic:08x}')

        meta.major = struct.unpack_from('<H', data, 4)[0]
        meta.minor = struct.unpack_from('<H', data, 6)[0]
        meta.header_size = struct.unpack_from('<I', data, 8)[0]
        meta._tables_size = struct.unpack_from('<I', data, 44)[0]

        def td(offset):
            return (struct.unpack_from('<I', data, offset)[0],
                    struct.unpack_from('<I', data, offset + 4)[0],
                    struct.unpack_from('<I', data, offset + 8)[0])

        p_off, p_cnt, p_sz = td(80)
        e_off, e_cnt, e_sz = td(92)
        g_off, g_cnt, g_sz = td(104)
        b_off, b_cnt, b_sz = td(116)

        base = meta.header_size

        # Partitions
        for i in range(p_cnt):
            o = base + p_off + i * p_sz
            name = data[o:o + 36].split(b'\x00')[0].decode('ascii')
            attrs = struct.unpack_from('<I', data, o + 36)[0]
            first_ext = struct.unpack_from('<I', data, o + 40)[0]
            num_ext = struct.unpack_from('<I', data, o + 44)[0]
            grp = struct.unpack_from('<I', data, o + 48)[0]
            meta.partitions.append(LPPartition(name, attrs, first_ext, num_ext, grp))

        # Extents
        for i in range(e_cnt):
            o = base + e_off + i * e_sz
            num_sec = struct.unpack_from('<Q', data, o)[0]
            ttype = struct.unpack_from('<I', data, o + 8)[0]
            tdata = struct.unpack_from('<Q', data, o + 12)[0]
            tsrc = struct.unpack_from('<I', data, o + 20)[0]
            meta.extents.append(LPExtent(num_sec, ttype, tdata, tsrc))

        # Groups
        for i in range(g_cnt):
            o = base + g_off + i * g_sz
            name = data[o:o + 36].split(b'\x00')[0].decode('ascii')
            flags = struct.unpack_from('<I', data, o + 36)[0]
            max_size = struct.unpack_from('<Q', data, o + 40)[0]
            meta.groups.append(LPGroup(name, flags, max_size))

        # Block devices
        for i in range(b_cnt):
            o = base + b_off + i * b_sz
            first_sec = struct.unpack_from('<Q', data, o)[0]
            align = struct.unpack_from('<I', data, o + 8)[0]
            align_off = struct.unpack_from('<I', data, o + 12)[0]
            size = struct.unpack_from('<Q', data, o + 16)[0]
            name = data[o + 24:o + 60].split(b'\x00')[0].decode('ascii')
            flags = struct.unpack_from('<I', data, o + 60)[0]
            meta.block_devices.append(LPBlockDevice(first_sec, align, align_off, size, name, flags))

        return meta

    def find_partition(self, name):
        for p in self.partitions:
            if p.name == name:
                return p
        return None

    def update_extent_sectors(self, extent_index, new_sectors):
        """Update the sector count of a specific extent and reserialize."""
        self.extents[extent_index].num_sectors = new_sectors
        # Update in raw data
        base = self.header_size
        e_off = struct.unpack_from('<I', self._raw, 92)[0]
        e_sz = struct.unpack_from('<I', self._raw, 100)[0]
        o = base + e_off + extent_index * e_sz
        struct.pack_into('<Q', self._raw, o, new_sectors)

    def serialize(self):
        """Return metadata bytes with recalculated checksums."""
        data = bytearray(self._raw[:self.header_size + self._tables_size])

        # Tables checksum (SHA256 of table data)
        tables = bytes(data[self.header_size:self.header_size + self._tables_size])
        tables_hash = hashlib.sha256(tables).digest()
        data[48:80] = tables_hash

        # Header checksum (SHA256 of header with checksum zeroed)
        data[12:44] = b'\x00' * 32
        header_hash = hashlib.sha256(bytes(data[:self.header_size])).digest()
        data[12:44] = header_hash

        return bytes(data)


# ---------------------------------------------------------------------------
# Super Partition I/O
# ---------------------------------------------------------------------------

class SuperPartition:
    def __init__(self, image_path):
        self.image_path = image_path
        self.gpt = GPTParser(image_path)
        self._super = self.gpt.find_partition('super')
        if not self._super:
            raise ValueError('No "super" partition found in GPT')
        self.super_offset = self._super.start_lba * SECTOR
        self.super_size = self._super.size_sectors * SECTOR
        self._geometry = None
        self._read_geometry()

    def _read_geometry(self):
        with open(self.image_path, 'rb') as f:
            f.seek(self.super_offset + 4096)
            geom = f.read(52)
            magic = struct.unpack_from('<I', geom, 0)[0]
            if magic != LP_METADATA_GEOMETRY_MAGIC:
                raise ValueError(f'Bad LP geometry magic: 0x{magic:08x}')
            self._geometry = {
                'metadata_max_size': struct.unpack_from('<I', geom, 40)[0],
                'metadata_slot_count': struct.unpack_from('<I', geom, 44)[0],
                'logical_block_size': struct.unpack_from('<I', geom, 48)[0],
            }

    def read_lp_metadata(self, slot=0):
        with open(self.image_path, 'rb') as f:
            offset = self.super_offset + 4096 * 3 + slot * self._geometry['metadata_max_size']
            f.seek(offset)
            data = f.read(self._geometry['metadata_max_size'])
            return LPMetadata.parse(data)

    def write_lp_metadata(self, metadata):
        """Write metadata to all primary and backup slots."""
        data = metadata.serialize()
        slot_count = self._geometry['metadata_slot_count']
        max_size = self._geometry['metadata_max_size']

        with open(self.image_path, 'r+b') as f:
            # Primary slots
            for slot in range(slot_count):
                offset = self.super_offset + 4096 * 3 + slot * max_size
                f.seek(offset)
                f.write(data)

            # Backup slots (before backup geometry at end of super)
            backup_geom_offset = self.super_offset + self.super_size - 4096 * 2
            for slot in range(slot_count):
                offset = backup_geom_offset - (slot_count - slot) * max_size
                f.seek(offset)
                f.write(data)

    def extract_partition_to_file(self, part_name, output_path):
        """Extract a logical partition image to a file."""
        meta = self.read_lp_metadata()
        part = meta.find_partition(part_name)
        if not part:
            available = [p.name for p in meta.partitions if p.num_extents > 0]
            raise ValueError(f'Partition "{part_name}" not found. Available: {", ".join(available)}')
        if part.num_extents == 0:
            raise ValueError(f'Partition "{part_name}" has no extents (empty)')

        with open(self.image_path, 'rb') as fin, open(output_path, 'wb') as fout:
            for i in range(part.num_extents):
                ext = meta.extents[part.first_extent + i]
                if ext.target_type == LPExtent.TARGET_LINEAR:
                    offset = self.super_offset + ext.target_data * SECTOR
                    fin.seek(offset)
                    remaining = ext.size_bytes
                    while remaining > 0:
                        chunk = min(remaining, 4 * 1024 * 1024)
                        fout.write(fin.read(chunk))
                        remaining -= chunk
                elif ext.target_type == LPExtent.TARGET_ZERO:
                    fout.write(b'\x00' * ext.size_bytes)

    def max_partition_size(self, part_name):
        """Calculate the maximum size a partition can grow to.

        For single-extent LINEAR partitions, this is the current extent size
        plus any free space immediately following it (up to the next extent
        or the end of the super partition).
        """
        meta = self.read_lp_metadata()
        part = meta.find_partition(part_name)
        if not part or part.num_extents == 0:
            return 0

        current_bytes = part.total_sectors(meta.extents) * SECTOR

        # Only handle growth for single-extent LINEAR partitions
        if part.num_extents != 1:
            return current_bytes
        ext = meta.extents[part.first_extent]
        if ext.target_type != LPExtent.TARGET_LINEAR:
            return current_bytes

        ext_end_sector = ext.target_data + ext.num_sectors

        # Find the nearest extent that starts after this one
        next_start = self.super_size // SECTOR  # end of super partition
        for i, other in enumerate(meta.extents):
            if i == part.first_extent:
                continue
            if other.num_sectors == 0 or other.target_type != LPExtent.TARGET_LINEAR:
                continue
            if other.target_data >= ext_end_sector:
                next_start = min(next_start, other.target_data)

        return next_start * SECTOR - ext.target_data * SECTOR

    def write_partition_from_file(self, part_name, input_path):
        """Write a partition image back into the super partition and update metadata."""
        meta = self.read_lp_metadata()
        part = meta.find_partition(part_name)
        if not part:
            raise ValueError(f'Partition "{part_name}" not found')
        if part.num_extents == 0:
            raise ValueError(f'Partition "{part_name}" has no extents')

        new_size = os.path.getsize(input_path)
        new_sectors = new_size // SECTOR

        # Calculate current total extent size
        old_total_sectors = part.total_sectors(meta.extents)
        old_total_bytes = old_total_sectors * SECTOR

        # If new image is larger, try to grow the extent into free space
        if new_size > old_total_bytes:
            max_size = self.max_partition_size(part_name)
            if new_size > max_size:
                raise ValueError(
                    f'New image ({new_size / 1024 / 1024:.1f} MB) exceeds '
                    f'available space ({max_size / 1024 / 1024:.1f} MB). '
                    f'Try more aggressive compression or reduce content.')
            # Grow the extent (single-extent case, validated by max_partition_size)
            ext = meta.extents[part.first_extent]
            ext.num_sectors = new_sectors
            old_total_sectors = new_sectors
            old_total_bytes = new_sectors * SECTOR

        with open(self.image_path, 'r+b') as fdisk, open(input_path, 'rb') as fin:
            written = 0
            for i in range(part.num_extents):
                ext = meta.extents[part.first_extent + i]
                if ext.target_type != LPExtent.TARGET_LINEAR:
                    continue
                offset = self.super_offset + ext.target_data * SECTOR
                fdisk.seek(offset)

                to_write = min(ext.size_bytes, new_size - written)
                if to_write > 0:
                    data = fin.read(to_write)
                    fdisk.write(data)
                    written += len(data)

                # Zero remaining space in this extent
                remaining_in_extent = ext.size_bytes - to_write
                if remaining_in_extent > 0:
                    fdisk.write(b'\x00' * remaining_in_extent)

        # Update extent sizes in metadata
        # For single-extent partitions (common case): just update the one extent
        if part.num_extents == 1:
            meta.update_extent_sectors(part.first_extent, new_sectors)
        else:
            # Multi-extent: fill extents in order, last one gets remainder
            remaining_sectors = new_sectors
            for i in range(part.num_extents):
                ext_idx = part.first_extent + i
                ext = meta.extents[ext_idx]
                if remaining_sectors >= ext.num_sectors:
                    remaining_sectors -= ext.num_sectors
                    # Keep this extent at its current size
                else:
                    meta.update_extent_sectors(ext_idx, remaining_sectors)
                    remaining_sectors = 0
                    # Zero out any subsequent extents
                    for j in range(i + 1, part.num_extents):
                        meta.update_extent_sectors(part.first_extent + j, 0)
                    break

        self.write_lp_metadata(meta)


# ---------------------------------------------------------------------------
# VBMeta
# ---------------------------------------------------------------------------

AVB_MAGIC = b'AVB0'
AVB_FLAGS_OFFSET = 120
AVB_FLAG_VERIFICATION_DISABLED = 2


class VBMeta:
    @staticmethod
    def read_flags(image_path, gpt):
        entry = gpt.find_partition('vbmeta_a')
        if not entry:
            return None, None
        offset = entry.start_lba * SECTOR
        with open(image_path, 'rb') as f:
            f.seek(offset)
            magic = f.read(4)
            if magic != AVB_MAGIC:
                return offset, None
            f.seek(offset + AVB_FLAGS_OFFSET)
            flags = struct.unpack('>I', f.read(4))[0]
            return offset, flags

    @staticmethod
    def disable_verity(image_path, gpt):
        offset, flags = VBMeta.read_flags(image_path, gpt)
        if offset is None:
            print('  Warning: vbmeta_a partition not found, skipping verity disable')
            return False
        if flags is None:
            print('  Warning: vbmeta_a does not have AVB magic, skipping')
            return False
        if flags & AVB_FLAG_VERIFICATION_DISABLED:
            print(f'  dm-verity already disabled (flags=0x{flags:08x})')
            return True

        with open(image_path, 'r+b') as f:
            f.seek(offset + AVB_FLAGS_OFFSET)
            f.write(struct.pack('>I', AVB_FLAG_VERIFICATION_DISABLED))

        print(f'  dm-verity disabled: flags 0x{flags:08x} -> 0x{AVB_FLAG_VERIFICATION_DISABLED:08x}')
        return True


# ---------------------------------------------------------------------------
# EROFS Tools Wrapper
# ---------------------------------------------------------------------------

class EROFSTools:
    @staticmethod
    def check_tools():
        for tool in ('mkfs.erofs', 'fsck.erofs'):
            if shutil.which(tool) is None:
                print(f'Error: {tool} not found. Install erofs-utils:', file=sys.stderr)
                print(f'  sudo apt-get install erofs-utils', file=sys.stderr)
                sys.exit(1)

    @staticmethod
    def extract(image_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        result = subprocess.run(
            ['fsck.erofs', f'--extract={output_dir}', image_path],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'fsck.erofs extract failed: {result.stderr}')

    @staticmethod
    def info(image_path):
        """Get EROFS image properties."""
        result = subprocess.run(
            ['dump.erofs', image_path],
            capture_output=True, text=True)
        props = {}
        for line in result.stdout.splitlines():
            if ':' in line:
                key, _, val = line.partition(':')
                props[key.strip()] = val.strip()
        # Also get from file command for quick magic check
        result2 = subprocess.run(['file', image_path], capture_output=True, text=True)
        props['_file_output'] = result2.stdout.strip()
        return props

    @staticmethod
    def build(source_dir, output_path, mount_point='/vendor',
              file_contexts=None, uuid=None, timestamp=None):
        cmd = [
            'mkfs.erofs',
            '-zlz4hc',
            '-C65536',
            '-b4096',
            f'--mount-point={mount_point}',
            '--all-root',
            '--force-gid=2000',
        ]
        if file_contexts and os.path.isfile(file_contexts):
            cmd.append(f'--file-contexts={file_contexts}')
        if uuid:
            cmd.append(f'-U{uuid}')
        if timestamp:
            cmd.append(f'-T{timestamp}')

        cmd.extend([output_path, source_dir])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f'mkfs.erofs failed: {result.stderr}')

        return os.path.getsize(output_path)


# ---------------------------------------------------------------------------
# Helper: find SELinux file_contexts in an extracted partition
# ---------------------------------------------------------------------------

def find_file_contexts(extract_dir, partition_name='vendor'):
    """Find the SELinux file_contexts file in an extracted partition."""
    # Look in etc/selinux/ for *_file_contexts
    selinux_dir = os.path.join(extract_dir, 'etc', 'selinux')
    if os.path.isdir(selinux_dir):
        # Prefer partition-specific file_contexts
        for pattern in [f'{partition_name}_file_contexts', '*_file_contexts']:
            matches = globmod.glob(os.path.join(selinux_dir, pattern))
            if matches:
                return matches[0]
    return None


def build_combined_file_contexts(sp, vendor_fc, work_dir):
    """Build combined file_contexts from plat (system_a) + vendor sources.

    Android uses a unified file_contexts when building EROFS images.  The
    platform file_contexts (from the system partition) contains critical
    catch-all rules like:
        /(vendor|system/vendor)(/.*)?   u:object_r:vendor_file:s0
    Without these, directories and files that aren't explicitly listed in
    vendor_file_contexts end up with NO SELinux label, which prevents boot.
    """
    combined_path = os.path.join(work_dir, 'combined_file_contexts')
    parts = []

    # --- Extract plat_file_contexts from system_a ---
    # Check for a cached copy next to the disk image first.
    cache_path = sp.image_path + '.plat_file_contexts'
    if os.path.isfile(cache_path):
        print(f'  Using cached platform file_contexts')
        with open(cache_path, 'r') as f:
            parts.append(f.read())
    else:
        meta = sp.read_lp_metadata()
        system_part = meta.find_partition('system_a') or meta.find_partition('system_b')
        if system_part and system_part.num_extents > 0:
            print(f'  Extracting platform file_contexts from {system_part.name} '
                  f'(one-time, will be cached)...')
            sys_img = os.path.join(work_dir, 'system.img')
            sys_dir = os.path.join(work_dir, 'system_extract')
            try:
                sp.extract_partition_to_file(system_part.name, sys_img)
                EROFSTools.extract(sys_img, sys_dir)

                # Expected path for system-as-root layout
                plat_fc = os.path.join(sys_dir, 'system', 'etc', 'selinux',
                                       'plat_file_contexts')
                if not os.path.isfile(plat_fc):
                    # Fallback: search the whole tree
                    for dp, _, fns in os.walk(sys_dir):
                        if 'plat_file_contexts' in fns:
                            plat_fc = os.path.join(dp, 'plat_file_contexts')
                            break
                    else:
                        plat_fc = None

                if plat_fc:
                    with open(plat_fc, 'r') as f:
                        content = f.read()
                    parts.append(content)
                    # Cache for future runs
                    with open(cache_path, 'w') as f:
                        f.write(content)
                    print(f'  Cached platform file_contexts for future runs')
                else:
                    print(f'  Warning: plat_file_contexts not found in system partition')
            except Exception as e:
                print(f'  Warning: could not extract platform file_contexts: {e}')
            finally:
                if os.path.exists(sys_img):
                    os.unlink(sys_img)
                if os.path.isdir(sys_dir):
                    shutil.rmtree(sys_dir)
        else:
            print(f'  Warning: system partition not found, using vendor file_contexts only')

    # --- Add vendor file_contexts ---
    if vendor_fc and os.path.isfile(vendor_fc):
        with open(vendor_fc, 'r') as f:
            parts.append(f.read())

    if not parts:
        return None

    with open(combined_path, 'w') as f:
        f.write('\n'.join(parts))

    return combined_path


def detect_mount_point(partition_name):
    """Infer the mount point from partition name."""
    base = partition_name.rstrip('_ab').removesuffix('_a').removesuffix('_b')
    mount_map = {
        'vendor': '/vendor',
        'system': '/system',
        'product': '/product',
        'system_ext': '/system_ext',
        'system_dlkm': '/system_dlkm',
        'vendor_dlkm': '/vendor_dlkm',
    }
    return mount_map.get(base, f'/{base}')


# ---------------------------------------------------------------------------
# Helper: backup
# ---------------------------------------------------------------------------

def ensure_backup(image_path, no_backup=False):
    if no_backup:
        return
    backup_path = image_path + '.bak'
    if os.path.exists(backup_path):
        print(f'  Backup already exists: {backup_path}')
        return
    print(f'  Creating backup: {backup_path}')
    shutil.copy2(image_path, backup_path)
    print(f'  Backup created ({os.path.getsize(backup_path) / 1024 / 1024 / 1024:.2f} GB)')


# ---------------------------------------------------------------------------
# Common repack logic (shared by update and repack commands)
# ---------------------------------------------------------------------------

def do_repack(image_path, partition_name, source_dir, no_backup=False):
    """Rebuild EROFS from source_dir and write back into the disk image."""
    EROFSTools.check_tools()

    sp = SuperPartition(image_path)
    meta = sp.read_lp_metadata()
    part = meta.find_partition(partition_name)
    if not part:
        available = [p.name for p in meta.partitions if p.num_extents > 0]
        print(f'Error: partition "{partition_name}" not found.', file=sys.stderr)
        print(f'Available: {", ".join(available)}', file=sys.stderr)
        sys.exit(1)

    old_total = part.total_sectors(meta.extents) * SECTOR
    max_total = sp.max_partition_size(partition_name)
    mount_point = detect_mount_point(partition_name)
    base_name = partition_name.removesuffix('_a').removesuffix('_b')

    # Temporary work directory for all intermediate files
    repack_work = tempfile.mkdtemp(prefix='super_tool_repack_')
    try:
        # --- SELinux file_contexts ---
        # Get partition-specific file_contexts from the extracted source
        vendor_fc = find_file_contexts(source_dir, base_name)

        # Build combined file_contexts (plat from system_a + vendor)
        # The plat_file_contexts has critical catch-all rules like:
        #   /(vendor|system/vendor)(/.*)?  u:object_r:vendor_file:s0
        file_contexts = build_combined_file_contexts(sp, vendor_fc, repack_work)
        if file_contexts:
            print(f'  Using combined file_contexts (plat + {base_name})')
        elif vendor_fc:
            file_contexts = vendor_fc
            print(f'  Warning: using {base_name}_file_contexts only (no plat)')
        else:
            print(f'  Warning: no file_contexts found, SELinux labels may be missing')

        # --- Get UUID from current partition ---
        uuid = None
        tmp_img = os.path.join(repack_work, 'current.img')
        sp.extract_partition_to_file(partition_name, tmp_img)
        props = EROFSTools.info(tmp_img)
        uuid = props.get('Filesystem UUID')
        if uuid:
            print(f'  Preserving UUID: {uuid}')
        os.unlink(tmp_img)

        # --- Build new EROFS image ---
        new_img = os.path.join(repack_work, 'new.img')
        print(f'  Building EROFS (LZ4HC)...')
        new_size = EROFSTools.build(
            source_dir, new_img,
            mount_point=mount_point,
            file_contexts=file_contexts,
            uuid=uuid)

        print(f'  New image: {new_size / 1024 / 1024:.1f} MB  '
              f'(limit: {max_total / 1024 / 1024:.1f} MB)')

        if new_size > old_total and new_size <= max_total:
            growth = (new_size - old_total) / 1024 / 1024
            print(f'  Growing partition by {growth:.1f} MB (free space available)')
        elif new_size > max_total:
            print(f'Error: new image is {(new_size - max_total) / 1024 / 1024:.1f} MB too large!',
                  file=sys.stderr)
            print(f'Try reducing content or using a different compression.', file=sys.stderr)
            sys.exit(1)

        # Backup
        ensure_backup(image_path, no_backup)

        # Write back
        print(f'  Writing partition...')
        sp.write_partition_from_file(partition_name, new_img)
        print(f'  LP metadata updated.')

        # Disable dm-verity
        print(f'  Disabling dm-verity...')
        VBMeta.disable_verity(image_path, sp.gpt)

    finally:
        if os.path.isdir(repack_work):
            shutil.rmtree(repack_work)


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

def cmd_info(args):
    sp = SuperPartition(args.image)
    meta = sp.read_lp_metadata()

    print(f'Disk image: {args.image}')
    print(f'Super partition: sector {sp._super.start_lba}, '
          f'size {sp.super_size / 1024 / 1024 / 1024:.1f} GB')
    print()

    # Groups
    print('Groups:')
    for i, g in enumerate(meta.groups):
        if g.name:
            print(f'  [{i}] {g.name:35s} max={g.max_size / 1024 / 1024:.0f} MB')
    print()

    # Partitions with extent info
    print(f'{"Partition":<22s} {"Size":>10s} {"Type":>6s} {"RO":>3s} {"Extents":>8s}  {"Physical offset"}')
    print('-' * 80)
    for p in meta.partitions:
        if p.num_extents == 0:
            size_str = '(empty)'
            type_str = ''
            offset_str = ''
        else:
            total = p.total_sectors(meta.extents)
            size_str = f'{total * SECTOR / 1024 / 1024:.1f} MB'
            ext = meta.extents[p.first_extent]

            # Detect filesystem type
            phys_offset = sp.super_offset + ext.target_data * SECTOR
            with open(args.image, 'rb') as f:
                f.seek(phys_offset + 1024)
                magic_bytes = f.read(4)
                magic = struct.unpack('<I', magic_bytes)[0]
                if magic == 0xE0F5E1E2:
                    type_str = 'EROFS'
                elif struct.unpack('<H', magic_bytes[0x38 - 0:0x38 + 2 - 0])[0] == 0xEF53 if len(magic_bytes) >= 0x3A else False:
                    type_str = 'ext4'
                else:
                    type_str = '?'

            offset_str = f'sector {ext.target_data} (disk sector {sp._super.start_lba + ext.target_data})'

        ro_str = 'RO' if p.readonly else 'RW'
        print(f'  {p.name:<20s} {size_str:>10s} {type_str:>6s} {ro_str:>3s} {p.num_extents:>8d}  {offset_str}')

    # VBMeta status
    print()
    _, flags = VBMeta.read_flags(args.image, sp.gpt)
    if flags is not None:
        if flags & AVB_FLAG_VERIFICATION_DISABLED:
            status = 'DISABLED'
        elif flags & 1:
            status = 'HASHTREE_DISABLED'
        else:
            status = 'ENABLED'
        print(f'dm-verity: {status} (vbmeta flags=0x{flags:08x})')
    else:
        print('dm-verity: unknown (vbmeta not found or invalid)')


# ---------------------------------------------------------------------------
# Subcommand: extract
# ---------------------------------------------------------------------------

def cmd_extract(args):
    sp = SuperPartition(args.image)
    output = args.output
    if not output:
        output = f'{args.partition}.img'

    print(f'Extracting {args.partition} -> {output}')
    sp.extract_partition_to_file(args.partition, output)
    size = os.path.getsize(output)
    print(f'Done: {size / 1024 / 1024:.1f} MB')


# ---------------------------------------------------------------------------
# Subcommand: dump
# ---------------------------------------------------------------------------

def cmd_dump(args):
    EROFSTools.check_tools()

    sp = SuperPartition(args.image)
    output_dir = args.output
    if not output_dir:
        output_dir = f'{args.partition}_dump'

    with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        print(f'Extracting {args.partition} image...')
        sp.extract_partition_to_file(args.partition, tmp_path)

        print(f'Dumping filesystem to {output_dir}...')
        EROFSTools.extract(tmp_path, output_dir)

        # Count files
        file_count = sum(len(files) for _, _, files in os.walk(output_dir))
        print(f'Done: {file_count} files extracted to {output_dir}')
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Subcommand: update
# ---------------------------------------------------------------------------

def cmd_update(args):
    EROFSTools.check_tools()

    sp = SuperPartition(args.image)

    with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as tmp:
        tmp_img = tmp.name
    work_dir = tempfile.mkdtemp(prefix='super_tool_')

    try:
        # Extract current partition
        print(f'Extracting {args.partition}...')
        sp.extract_partition_to_file(args.partition, tmp_img)

        # Dump contents
        print(f'Dumping filesystem...')
        EROFSTools.extract(tmp_img, work_dir)
        os.unlink(tmp_img)

        # Overlay files
        dest_dir = work_dir
        if args.dest:
            dest_dir = os.path.join(work_dir, args.dest)
            os.makedirs(dest_dir, exist_ok=True)

        overlay = os.path.abspath(args.overlay)
        file_count = 0
        if os.path.isfile(overlay):
            dst = os.path.join(dest_dir, os.path.basename(overlay))
            shutil.copy2(overlay, dst)
            file_count = 1
        else:
            for root, dirs, files in os.walk(overlay):
                rel = os.path.relpath(root, overlay)
                target = os.path.join(dest_dir, rel) if rel != '.' else dest_dir
                os.makedirs(target, exist_ok=True)
                for fname in files:
                    src = os.path.join(root, fname)
                    dst = os.path.join(target, fname)
                    shutil.copy2(src, dst)
                    file_count += 1
        print(f'  Overlaid {file_count} files' +
              (f' into {args.dest}/' if args.dest else ''))

        # Repack
        do_repack(args.image, args.partition, work_dir, no_backup=args.no_backup)

    finally:
        if os.path.isdir(work_dir):
            shutil.rmtree(work_dir)

    print(f'\nDone! {args.partition} updated successfully.')


# ---------------------------------------------------------------------------
# Subcommand: repack
# ---------------------------------------------------------------------------

def cmd_repack(args):
    source_dir = os.path.abspath(args.source_dir)
    if not os.path.isdir(source_dir):
        print(f'Error: {source_dir} is not a directory', file=sys.stderr)
        sys.exit(1)

    print(f'Repacking {args.partition} from {source_dir}...')
    do_repack(args.image, args.partition, source_dir, no_backup=args.no_backup)
    print(f'\nDone! {args.partition} repacked successfully.')


# ---------------------------------------------------------------------------
# Subcommand: restore
# ---------------------------------------------------------------------------

def cmd_restore(args):
    backup_path = args.image + '.bak'
    if not os.path.exists(backup_path):
        print(f'Error: backup not found: {backup_path}', file=sys.stderr)
        sys.exit(1)

    print(f'Restoring {args.image} from {backup_path}...')
    shutil.copy2(backup_path, args.image)
    print(f'Done! Image restored ({os.path.getsize(args.image) / 1024 / 1024 / 1024:.2f} GB)')


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Android Super Partition Toolkit',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  %(prog)s info image.bin
  %(prog)s extract image.bin vendor_a -o vendor.img
  %(prog)s dump image.bin vendor_a -o vendor_work/
  %(prog)s update image.bin vendor_a ./firmware/ --dest firmware/
  %(prog)s repack image.bin vendor_a vendor_work/
  %(prog)s restore image.bin
""")

    sub = parser.add_subparsers(dest='command', required=True)

    # info
    p = sub.add_parser('info', help='Show partition layout and dm-verity status')
    p.add_argument('image', help='Disk image file')
    p.set_defaults(func=cmd_info)

    # extract
    p = sub.add_parser('extract', help='Extract raw partition image')
    p.add_argument('image', help='Disk image file')
    p.add_argument('partition', help='Logical partition name (e.g., vendor_a)')
    p.add_argument('-o', '--output', help='Output file (default: <partition>.img)')
    p.set_defaults(func=cmd_extract)

    # dump
    p = sub.add_parser('dump', help='Extract partition contents to directory')
    p.add_argument('image', help='Disk image file')
    p.add_argument('partition', help='Logical partition name (e.g., vendor_a)')
    p.add_argument('-o', '--output', help='Output directory (default: <partition>_dump)')
    p.set_defaults(func=cmd_dump)

    # update
    p = sub.add_parser('update', help='Add/update files in a partition')
    p.add_argument('image', help='Disk image file')
    p.add_argument('partition', help='Logical partition name (e.g., vendor_a)')
    p.add_argument('overlay', help='File or directory to add/overlay')
    p.add_argument('--dest', help='Destination subdirectory within partition (e.g., firmware/)')
    p.add_argument('--no-backup', action='store_true', help='Skip automatic backup')
    p.set_defaults(func=cmd_update)

    # repack
    p = sub.add_parser('repack', help='Rebuild partition from modified directory')
    p.add_argument('image', help='Disk image file')
    p.add_argument('partition', help='Logical partition name (e.g., vendor_a)')
    p.add_argument('source_dir', help='Directory containing full partition contents')
    p.add_argument('--no-backup', action='store_true', help='Skip automatic backup')
    p.set_defaults(func=cmd_repack)

    # restore
    p = sub.add_parser('restore', help='Restore disk image from .bak backup')
    p.add_argument('image', help='Disk image file')
    p.set_defaults(func=cmd_restore)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
