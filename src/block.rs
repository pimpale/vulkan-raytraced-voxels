use std::{io, fs};
use std::collections::HashMap;

const IMG_HEIGHT: i32 = 16;
const IMG_WIDTH: i32 = 16;

pub enum BlockFaceKind {
  Down = 1,
  Up = 2,
  Left = 3,
  Right = 4,
  Back = 5,
  Front = 6,
}

static void writePicTexAtlas(                       //
    uint8_t pTextureAtlas[BLOCK_TEXTURE_ATLAS_LEN], //
    BlockIndex index,                               //
    BlockFaceKind face,                             //
    const char *assetPath                           //
) {
  const char *blockName = BLOCKS[index].name;

  const char *faceFilename;
  switch (face) {
  case Block_DOWN:
    faceFilename = "down.ff";
    break;
  case Block_UP:
    faceFilename = "up.ff";
    break;
  case Block_LEFT:
    faceFilename = "left.ff";
    break;
  case Block_RIGHT:
    faceFilename = "right.ff";
    break;
  case Block_BACK:
    faceFilename = "back.ff";
    break;
  case Block_FRONT:
    faceFilename = "front.ff";
    break;
  }

  // get size of total buffer
  size_t fileNameSize = strlen(assetPath) + strlen("/") + strlen(blockName) +
                        strlen("/") + strlen(faceFilename) + 1;
  char *fileName = malloc(fileNameSize * sizeof(char));

  // build string
  strcpy(fileName, assetPath);
  strcat(fileName, "/");
  strcat(fileName, blockName);
  strcat(fileName, "/");
  strcat(fileName, faceFilename);

  farbfeld_img img;
  farbfeld_error e = read_farbfeld_img(&img, fileName);
  if (e != farbfeld_OK) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "could not open farbfeld file at: %s", fileName);
    PANIC();
  }

  // assert dimensions of image
  if(img.xsize != BLOCK_TEXTURE_SIZE || img.ysize != BLOCK_TEXTURE_SIZE) {
    LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "expected dimensions %u by %u: %s", BLOCK_TEXTURE_SIZE, BLOCK_TEXTURE_SIZE, fileName);
    PANIC();
  }

  free(fileName);

  // now write to area
  overwriteRectBmp(               //
      pTextureAtlas,              //
      BLOCK_TEXTURE_ATLAS_WIDTH,  //
      BLOCK_TEXTURE_ATLAS_HEIGHT, //
      face * BLOCK_TEXTURE_SIZE,  //
      index * BLOCK_TEXTURE_SIZE, //
      &img                        //
  );
  free_farbfeld_img(&img);
}

pub fn build_texture_atlas<P>(dir:P) -> Result<HashMap<String, BlockTextures>, io::Error> {

  for maybe_block_path in fs::read_dir(dir)? {
        let block_path = block_path?;
        

  }


  for (BlockIndex i = 0; i < BLOCKS_LEN; i++) {
    // don't need to get texture for transparent blocks
    if (BLOCKS[i].transparent) {
      continue;
    }

    // write all six faces
    writePicTexAtlas(pTextureAtlas, i, Block_DOWN, assetPath);
    writePicTexAtlas(pTextureAtlas, i, Block_UP, assetPath);
    writePicTexAtlas(pTextureAtlas, i, Block_LEFT, assetPath);
    writePicTexAtlas(pTextureAtlas, i, Block_RIGHT, assetPath);
    writePicTexAtlas(pTextureAtlas, i, Block_BACK, assetPath);
    writePicTexAtlas(pTextureAtlas, i, Block_FRONT, assetPath);
  }
}
