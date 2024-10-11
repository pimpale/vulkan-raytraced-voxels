# Vulkan Raytraced Voxel Renderer

Some pictures:

### 8 samples per pixel, 4 bounces (28 fps)
![8 samples per pixel, 4 bounces, 128x128x128 voxel grid](./assets/screenshots/8spp_800x600.png)

### 128 samples per pixel, 4 bounces (3 fps)
![128 samples per pixel, 4 bounces, 128x128x128 voxel grid](./assets/screenshots/128spp_800x600.png)

## How to build

Install shaderc and the Vulkan SDK.

Then:
```bash
cargo build --release
```

## How to modify image assets
Block textures can be found in the `assets/blocks/` directory.
