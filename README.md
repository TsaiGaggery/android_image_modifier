# super_tool.py — Android Super Partition Toolkit

A Python tool to inspect, extract, modify, and repack logical partitions inside
Android GPT disk images that use dynamic partitions (super).

Built for modifying EROFS-based partitions (vendor, system, product, etc.) in
full disk images — for example, adding firmware blobs or kernel modules to a
vendor partition without a full AOSP rebuild.

## Requirements

- **Python 3.10+** (no external Python dependencies — stdlib only)
- **erofs-utils** — provides `mkfs.erofs`, `fsck.erofs`, and `dump.erofs`

```bash
sudo apt-get install erofs-utils
```

## Quick start

```bash
# See what's inside the image
./super_tool.py info image.bin

# Add WiFi firmware to the vendor partition (one command)
./super_tool.py update image.bin vendor_a ./linux-firmware/ --dest firmware/

# Add kernel modules to vendor_dlkm
./super_tool.py update image.bin vendor_dlkm_a ./my-modules/ --dest lib/modules/
```

## Commands

### info

Show the partition layout, sizes, filesystem types, and dm-verity status.

```
./super_tool.py info <disk_image>
```

Example output:

```
Disk image: android-desktop_image-firefly-use.bin
Super partition: sector 475136, size 8.0 GB

Groups:
  [0] default                             max=0 MB
  [1] google_dynamic_partitions_a         max=3800 MB
  [2] google_dynamic_partitions_b         max=3800 MB

Partition                    Size   Type  RO  Extents  Physical offset
--------------------------------------------------------------------------------
  system_a               769.8 MB  EROFS  RO        1  sector 2048
  vendor_a               602.0 MB  EROFS  RO        1  sector 1579008
  product_a              700.7 MB  EROFS  RO        1  sector 2850816
  system_ext_a           138.2 MB  EROFS  RO        1  sector 4286464
  system_dlkm_a            8.4 MB  EROFS  RO        1  sector 4571136
  vendor_dlkm_a           19.5 MB  EROFS  RO        1  sector 4589568

dm-verity: DISABLED (vbmeta flags=0x00000002)
```

### extract

Extract a raw partition image (EROFS) to a file.

```
./super_tool.py extract <disk_image> <partition> [-o output_file]
```

```bash
./super_tool.py extract image.bin vendor_a -o vendor.img
# vendor.img is now a standalone EROFS image you can mount or inspect
```

### dump

Extract the full filesystem contents of a partition to a directory.

```
./super_tool.py dump <disk_image> <partition> [-o output_dir]
```

```bash
./super_tool.py dump image.bin vendor_a -o vendor_work/
ls vendor_work/
# apex  bin  build.prop  etc  firmware  lib  lib64  ...
```

### update

Add or update files inside a partition in a single step. This is the main
command for most use cases.

```
./super_tool.py update <disk_image> <partition> <overlay_dir> [--dest <subdir>] [--no-backup]
```

The tool will:
1. Create a backup of the disk image (`.bak`, skipped if one already exists)
2. Extract the current partition contents
3. Copy your overlay files on top (into `--dest` subdirectory if specified)
4. Build combined SELinux `file_contexts` (platform + partition-specific)
5. Rebuild the EROFS image with LZ4HC compression and proper SELinux labels
6. Write it back into the super partition
7. Update LP metadata with recalculated checksums
8. Disable dm-verity in vbmeta

```bash
# Add firmware files into vendor/firmware/
./super_tool.py update image.bin vendor_a ./linux-firmware/ --dest firmware/

# Overlay files directly (paths mirror the partition root)
./super_tool.py update image.bin vendor_a ./my-overlay/
# Files in my-overlay/firmware/foo.bin -> vendor/firmware/foo.bin
# Files in my-overlay/etc/config.xml -> vendor/etc/config.xml
```

### repack

Rebuild a partition from a directory you previously dumped and modified manually.

```
./super_tool.py repack <disk_image> <partition> <source_dir> [--no-backup]
```

This is for the manual workflow where you want full control over the contents:

```bash
# Step 1: dump to a working directory
./super_tool.py dump image.bin vendor_a -o vendor_work/

# Step 2: make your changes
cp my-firmware/* vendor_work/firmware/
rm vendor_work/firmware/old-unused.bin
vim vendor_work/etc/some-config.xml

# Step 3: repack
./super_tool.py repack image.bin vendor_a vendor_work/
```

### restore

Restore the disk image from the `.bak` backup created by `update` or `repack`.

```
./super_tool.py restore <disk_image>
```

```bash
./super_tool.py restore image.bin
# Restores image.bin from image.bin.bak
```

## How it works

Android disk images use a **super** partition that contains multiple logical
partitions (system, vendor, product, etc.) managed by
[LP metadata](https://source.android.com/docs/core/ota/dynamic_partitions)
(Android Dynamic Partitions). Each logical partition is typically an
[EROFS](https://erofs.docs.kernel.org/) read-only compressed filesystem.

This tool handles the full pipeline:

```
GPT disk image
  └── super partition (8 GB)
        ├── LP metadata (geometry + partition table with checksums)
        ├── system_a    (EROFS)
        ├── vendor_a    (EROFS)  ← we modify this
        ├── product_a   (EROFS)
        └── ...
```

1. **Parse GPT** to locate the super partition
2. **Parse LP metadata** to find logical partitions and their physical extents
3. **Extract EROFS** image from the super partition
4. **Dump filesystem** contents using `fsck.erofs`
5. **Modify** the extracted tree (add/update/remove files)
6. **Rebuild EROFS** with `mkfs.erofs` using LZ4HC compression
7. **Apply SELinux labels** using combined `file_contexts` (see below)
8. **Write back** into the super partition, zeroing unused space
9. **Update LP metadata** in all primary and backup slots with recalculated SHA256 checksums
10. **Disable dm-verity** by setting vbmeta flags (required for modified partitions to boot)

## SELinux labels

Correct SELinux labeling is critical — missing labels will prevent Android from
booting. The tool automatically builds a **combined** `file_contexts` by
extracting and merging:

- **`plat_file_contexts`** from the `system_a` partition — contains catch-all
  rules like `/(vendor|system/vendor)(/.*)?  u:object_r:vendor_file:s0` that
  label the partition root and any paths not explicitly listed elsewhere
- **`vendor_file_contexts`** (or the target partition's own `*_file_contexts`)
  — contains partition-specific rules for firmware, binaries, libraries, etc.

Both are required. Using `vendor_file_contexts` alone leaves the root directory
and common subdirectories unlabeled, which causes boot failure.

The `plat_file_contexts` is extracted from `system_a` on first use and cached
alongside the disk image (as `<image>.plat_file_contexts`) so subsequent runs
don't need to re-extract the system partition.

## dm-verity and Verified Boot

Modifying any partition breaks dm-verity because the Merkle hash tree no longer
matches the data. The tool automatically disables verification by setting the
`AVB_VBMETA_IMAGE_FLAGS_VERIFICATION_DISABLED` flag in vbmeta.

**Requirements:**
- The device bootloader must be **unlocked**
- A yellow/orange warning screen will appear at boot — this is normal

## Size constraints

The rebuilt EROFS image must fit within the original partition's allocated space
in the super partition. The tool uses LZ4HC compression (better ratio than
standard LZ4, fully compatible for decompression) to minimize image size.

If the new image exceeds the available space, the tool will report an error with
the size difference. Options in that case:
- Remove unnecessary files from the partition
- Check if you're adding duplicate or oversized content

Use `info` to see current partition sizes and the dynamic partition group limit
to understand available headroom.

## Supported partition types

| Filesystem | Read | Write |
|------------|------|-------|
| EROFS      | Yes  | Yes   |
| ext4       | No   | No    |

Currently only EROFS partitions are supported for modification (this covers
most modern Android images). ext4 support could be added if needed.

## Common use cases

**Add WiFi/Bluetooth firmware:**
```bash
./super_tool.py update image.bin vendor_a ./linux-firmware/ --dest firmware/
```

**Add kernel modules:**
```bash
./super_tool.py update image.bin vendor_dlkm_a ./modules/ --dest lib/modules/
```

**Update a config file:**
```bash
mkdir -p overlay/etc
cp modified-config.xml overlay/etc/
./super_tool.py update image.bin vendor_a overlay/
```

**Inspect what's in a partition:**
```bash
./super_tool.py dump image.bin system_a -o system_contents/
find system_contents/ -name "*.apk" | head
```
