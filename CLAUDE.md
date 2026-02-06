# CLAUDE.md — Project Guide

## What this project is

`super_tool.py` is a Python CLI tool for modifying logical partitions (vendor, system, product, etc.) inside Android GPT disk images that use dynamic partitions (super). It handles the full pipeline: GPT parsing, LP metadata, EROFS extract/rebuild, SELinux labeling, and dm-verity disabling.

## Key files

- `super_tool.py` — the main tool (~980 lines, stdlib-only Python 3.10+)
- `README.md` — user-facing documentation
- `android-desktop_image-firefly-use.bin` — the Android disk image being modified
- `android-desktop_image-firefly-use.bin.bak` — backup of the original image
- `android-desktop_image-firefly-use.bin.plat_file_contexts` — cached platform SELinux contexts (extracted from system_a)
- `linux-firmware/` — firmware files to overlay into vendor/firmware/

## External dependencies

- `erofs-utils` package: provides `mkfs.erofs`, `fsck.erofs`, `dump.erofs`
- User does not have sudo — ask them to install packages manually

## Critical technical details

### SELinux file_contexts (boot-critical)

EROFS rebuild MUST use combined `plat_file_contexts` (from system_a) + `vendor_file_contexts`. The platform file has catch-all rules like `/(vendor|system/vendor)(/.*)?  u:object_r:vendor_file:s0`. Without these, root `/` and `/bin` get NO SELinux label (Xattr size=0) and the device will not boot. This was the root cause of two boot failures before being fixed.

### EROFS rebuild flags

Current mkfs.erofs invocation uses:
- `-zlz4hc` — LZ4HC compression (better ratio, LZ4-compatible decompression)
- `-C65536` — cluster size
- `-b4096` — block size
- `--mount-point=/vendor` — so file_contexts paths match correctly
- `--all-root --force-gid=2000` — sets uid=0, gid=2000 for all entries
- `--file-contexts=<combined>` — applies SELinux labels

The `--all-root --force-gid=2000` is a simplification. The actual Android build uses `--fs-config-file` for per-file UID/GID/mode (dirs get 0:2000, regular files get 0:0, bin executables get 0:2000). This hasn't caused boot issues but could be improved for full correctness.

### A/B slots

The image uses A/B partitioning. Only slot A has data; slot B partitions are empty. Only modify `_a` partitions.

### LP metadata

- 3 primary slots + 3 backup slots at end of super partition
- SHA256 checksums for both header (with checksum field zeroed) and tables
- Must update ALL slots when changing partition extent sizes

### dm-verity

- vbmeta flags at offset 120 (big-endian uint32): 0=enabled, 2=disabled
- Must be disabled after any partition modification
- Requires unlocked bootloader

## Running the tool

```bash
# Show partition layout
python3 super_tool.py info android-desktop_image-firefly-use.bin

# Add firmware (the main use case)
python3 super_tool.py update android-desktop_image-firefly-use.bin vendor_a ./linux-firmware/ --dest firmware/

# Restore from backup if something goes wrong
python3 super_tool.py restore android-desktop_image-firefly-use.bin
```

## Testing changes

After modifying the script, verify with:
1. `python3 -c "import ast; ast.parse(open('super_tool.py').read())"` — syntax check
2. `python3 super_tool.py info android-desktop_image-firefly-use.bin` — smoke test
3. Extract the rebuilt partition and check with `dump.erofs --path=/ <image>` that Xattr size is 16 (not 0) on root and /bin
