# Vulkan Raytraced Voxel Renderer

Not done yet.

## How to build

Before you can run the program, you need to ensure that you have installed the Vulkan libraries and headers.

```bash
$ cd assets/shaders
$ ./compile.sh
$ cd ..
$ make
```

Run from the project root directory.

```bash
$ ./obj/vulkan-triangle-v2
```

## How to modify image assets
Block textures can be found in the `assets/blocks/` directory.
These textures are in the [farbfeld](http://tools.suckless.org/farbfeld/) format.
If you want to edit these files, you can use ImageMagick tools to convert them to png format,
and then back again to farbfeld format once you are done editing.

Example:
```bash
$ convert up.ff up.png
```

