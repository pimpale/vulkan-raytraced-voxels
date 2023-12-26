use std::fmt::Display;

use image::Rgba32FImage;

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

    // cached block texture indices
    air_idx: Option<usize>,
    grass_idx: Option<usize>,
    soil_idx: Option<usize>,
    stone_idx: Option<usize>,
}

impl BlockDefinitionTable {
    // appends block textures to current_texture_atlas
    pub fn load_assets(
        assets_path: &str,
        mut current_texture_atlas: Vec<Rgba32FImage>,
    ) -> BlockDefinitionTable {
        let block_textures_offset = current_texture_atlas.len();

        let blocks: Vec<BlockDefinition> = std::fs::read_dir(format!("{}/blocks", assets_path))
            .unwrap()
            .map(|x| x.unwrap())
            .map(|x| {
                let name = x.file_name().into_string().unwrap();
                // single file is not a block
                let transparent = !x.file_type().unwrap().is_file();
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
                let texture_path = format!("{}/blocks/{}/{}.png", assets_path, block.name, face);
                let texture = image::open(texture_path).unwrap().to_rgba32f();
                current_texture_atlas.push(texture);
            }
        }

        let air_idx = blocks
            .iter()
            .position(|x| x.name == "air");
        let grass_idx = blocks
            .iter()
            .position(|x| x.name == "grass");
        let soil_idx = blocks
            .iter()
            .position(|x| x.name == "soil");
        let stone_idx = blocks
            .iter()
            .position(|x| x.name == "stone");

        BlockDefinitionTable {
            block_textures_offset,
            blocks,
            air_idx,
            grass_idx,
            soil_idx,
            stone_idx,
        }
    }

    pub fn get_texture_offset(&self, block_idx: usize, face: BlockFace) -> u32 {
        let texture_idx = self.block_textures_offset + block_idx * 6 + face as usize;
        texture_idx as u32
    }

    pub fn transparent(&self, block_idx: usize) -> bool {
        self.blocks[block_idx].transparent
    }

    pub fn air_idx(&self) -> Option<usize> {
        self.air_idx
    }

    pub fn grass_idx(&self) -> Option<usize> {
        self.grass_idx
    }

    pub fn soil_idx(&self) -> Option<usize> {
        self.soil_idx
    }

    pub fn stone_idx(&self) -> Option<usize> {
        self.stone_idx
    }

}
