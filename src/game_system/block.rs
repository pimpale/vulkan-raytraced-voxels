use std::fmt::Display;

use image::RgbaImage;

#[derive(Copy, Clone)]
pub enum BlockFace {
    DOWN,
    UP,
    LEFT,
    RIGHT,
    BACK,
    FRONT,
}

impl Display for BlockFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockFace::DOWN => write!(f, "down"),
            BlockFace::UP => write!(f, "up"),
            BlockFace::LEFT => write!(f, "left"),
            BlockFace::RIGHT => write!(f, "right"),
            BlockFace::BACK => write!(f, "back"),
            BlockFace::FRONT => write!(f, "front"),
        }
    }
}

pub struct BlockDefinition {
    pub name: String,
    pub transparent: bool,
}

pub struct BlockDefinitionTable {
    block_textures_offset: usize,
    blocks: Vec<BlockDefinition>,
}

pub type BlockIdx = u8;

impl BlockDefinitionTable {
    // appends block textures to current_texture_atlas
    pub fn load_assets(
        assets_path: &str,
        current_texture_atlas: &mut Vec<RgbaImage>,
    ) -> BlockDefinitionTable {
        let block_textures_offset = current_texture_atlas.len();

        let blocks: Vec<BlockDefinition> = std::fs::read_dir(assets_path)
            .unwrap()
            .map(|x| x.unwrap())
            .map(|x| {
                let name = x.file_name().into_string().unwrap();
                // single file is not a block
                let transparent = x.file_type().unwrap().is_file();
                BlockDefinition { name, transparent }
            })
            .collect();

        for block in &blocks {
            if block.transparent {
                continue;
            }

            for face in [
                BlockFace::DOWN,
                BlockFace::UP,
                BlockFace::LEFT,
                BlockFace::RIGHT,
                BlockFace::BACK,
                BlockFace::FRONT,
            ] {
                let texture_path = format!("{}/{}/{}.ff", assets_path, block.name, face);
                let texture = image::open(texture_path).unwrap().to_rgba8();
                current_texture_atlas.push(texture);
            }
        }
        BlockDefinitionTable {
            block_textures_offset,
            blocks,
        }
    }

    pub fn get_texture_offset(&self, block_idx: BlockIdx, face: BlockFace) -> u32 {
        let texture_idx = self.block_textures_offset + (block_idx as usize) * 6 + face as usize;
        texture_idx as u32
    }

    pub fn transparent(&self, block_idx: BlockIdx) -> bool {
        self.blocks[block_idx as usize].transparent
    }

    pub fn block_idx(&self, name: &str) -> Option<BlockIdx> {
        self.blocks
            .iter()
            .position(|x| x.name == name)
            .map(|x| x as BlockIdx)
    }
}
