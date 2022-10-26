use std::{io, fs};

pub struct BlockDef {
    pub transparent: bool,
    pub name: &'static str,
}

const DEF_LIST: [BlockDef;5] = [
    BlockDef { transparent: true, name: "air",},
    BlockDef { transparent: true, name: "grass",},
    BlockDef { transparent: true, name: "stone",},
    BlockDef { transparent: true, name: "soil",},
];

const FACE_IMG_XSIZE: i32 = 16;
const FACE_IMG_YSIZE: i32 = 16;

const ATLAS_IMG_XSIZE: i32 = FACE_IMG_XSIZE * 6;
const ATLAS_IMG_YSIZE: i32 = FACE_IMG_YSIZE * DEF_LIST.len();

const TILE_TEX_XSIZE:f32 = 1.0/6.0;
const TILE_TEX_YSIZE:f32 = 1.0/(DEF_LIST.len() as f32);

pub enum BlockFaceKind {
  Down = 1,
  Up = 2,
  Left = 3,
  Right = 4,
  Back = 5,
  Front = 6,
}


fn write_pic_tex_atlas<P>(
    &mut atlas: image::GenericImage,
    path: P,
    block_name: &str,
    face: BlockFaceKind,
) -> Result<(), image::ImageError>{
  path.join(match face {
    BlockFaceKind::Down => "down.ff",
    BlockFaceKind::Up => "up.ff",
    BlockFaceKind::Left => "left.ff",
    BlockFaceKind::Right => "right.ff",
    BlockFaceKind::Back => "back.ff",
    BlockFaceKind::Front => "front.ff",
  });

  let si = atlas.sub_image(0, face*FACE_IMG_YSIZE, FACE_IMG_XSIZE, FACE_IMG_YSIZE);
  si.copy_from(image::load(pat));

  // farbfeld_error e = read_farbfeld_img(&img, fileName);
  // if (e != farbfeld_OK) {
  //   LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "could not open farbfeld file at: %s", fileName);
  //   PANIC();
  // }

  // // assert dimensions of image
  // if(img.xsize != BLOCK_TEXTURE_SIZE || img.ysize != BLOCK_TEXTURE_SIZE) {
  //   LOG_ERROR_ARGS(ERR_LEVEL_FATAL, "expected dimensions %u by %u: %s", BLOCK_TEXTURE_SIZE, BLOCK_TEXTURE_SIZE, fileName);
  //   PANIC();
  // }


  // // now write to area
  // overwriteRectBmp(               //
  //     pTextureAtlas,              //
  //     BLOCK_TEXTURE_ATLAS_WIDTH,  //
  //     BLOCK_TEXTURE_ATLAS_HEIGHT, //
  //     face * IMG_XSIZE,  //
  //     index * IMG_YSIZE, //
  //     &img                        //
  // );
  // free_farbfeld_img(&img);

  return Ok(());
}

pub fn build_texture_atlas<P>(dir:P) -> Result<RgbaImage, io::Error> {
  let mut atlas = RgbaImage::new(32, 32);
  for BlockDef {transparent, name} in DEF_LIST{
      if !transparent {
          let dir

    // write all six faces
    write_pic_tex_atlas(pTextureAtlas, Block_DOWN, assetPath);
    write_pic_tex_atlas(pTextureAtlas, Block_UP, assetPath);
    write_pic_tex_atlas(pTextureAtlas, Block_LEFT, assetPath);
    write_pic_tex_atlas(pTextureAtlas, Block_RIGHT, assetPath);
    write_pic_tex_atlas(pTextureAtlas, Block_BACK, assetPath);
    write_pic_tex_atlas(pTextureAtlas, Block_FRONT, assetPath);
      }
  }
}
