use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap},
    fmt::Display,
};

use image::RgbImage;

#[derive(Copy, Clone, Debug)]
pub enum BlockFace {
    LEFT,
    RIGHT,
    DOWN,
    UP,
    BACK,
    FRONT,
}

impl Display for BlockFace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockFace::LEFT => write!(f, "left"),
            BlockFace::RIGHT => write!(f, "right"),
            BlockFace::DOWN => write!(f, "down"),
            BlockFace::UP => write!(f, "up"),
            BlockFace::BACK => write!(f, "back"),
            BlockFace::FRONT => write!(f, "front"),
        }
    }
}

#[derive(Deserialize)]
pub struct TextureDefinition {
    // texture for how much light is reflected at each point
    reflectivity: String,
    // texture for how much light is emitted at each point
    emissivity: String,
    // texture for the chance that the the light will be reflected specularly at each point
    metallicity: String,
}

#[derive(Deserialize)]
pub struct BlockJson {
    pub left: TextureDefinition,
    pub right: TextureDefinition,
    pub down: TextureDefinition,
    pub up: TextureDefinition,
    pub back: TextureDefinition,
    pub front: TextureDefinition,
}

#[derive(Deserialize)]
pub struct BlocksJson {
    pub blocks: BTreeMap<String, BlockJson>,
}

pub struct BlockDefinitionTable {
    block_textures_offset: usize,
    block_lookup: HashMap<String, BlockIdx>,
}

pub type BlockIdx = u8;

impl BlockDefinitionTable {
    // appends block textures to current_texture_atlas
    pub fn load_assets(
        assets_path: &str,
        current_texture_atlas: &mut Vec<(RgbImage, RgbImage, RgbImage)>,
    ) -> BlockDefinitionTable {
        let block_textures_offset = current_texture_atlas.len();

        let block_definitions_path = format!("{}/blocks.json", assets_path);
        let blocks_json: BlocksJson =
            serde_json::from_str(&std::fs::read_to_string(block_definitions_path).unwrap())
                .unwrap();

        let mut block_lookup = HashMap::new();
        block_lookup.insert("air".to_string(), blocks_json.blocks.len() as BlockIdx);

        for (idx, (name, block)) in blocks_json.blocks.into_iter().enumerate() {
            let mut load_texture = |tex: &TextureDefinition| {
                let reflectivity_path = format!("{}/{}", assets_path, tex.reflectivity);
                let reflectivity = image::open(reflectivity_path).unwrap().to_rgb8();
                let emissivity_path = format!("{}/{}", assets_path, tex.emissivity);
                let emissivity = image::open(emissivity_path).unwrap().to_rgb8();
                let metallicity_path = format!("{}/{}", assets_path, tex.metallicity);
                let metallicity = image::open(metallicity_path).unwrap().to_rgb8();
                current_texture_atlas.push((reflectivity, emissivity, metallicity));
            };

            load_texture(&block.left);
            load_texture(&block.right);
            load_texture(&block.down);
            load_texture(&block.up);
            load_texture(&block.back);
            load_texture(&block.front);

            block_lookup.insert(name, idx as BlockIdx);
        }

        BlockDefinitionTable {
            block_textures_offset,
            block_lookup,
        }
    }

    pub fn get_material_offset(&self, block_idx: BlockIdx, face: BlockFace) -> u32 {
        let texture_idx = self.block_textures_offset + (block_idx as usize) * 6 + face as usize;
        texture_idx as u32
    }

    pub fn transparent(&self, block_idx: BlockIdx) -> bool {
        block_idx == self.block_lookup.len() as BlockIdx - 1
    }

    pub fn block_idx(&self, name: &str) -> Option<BlockIdx> {
        self.block_lookup.get(name).map(|x| *x)
    }
}
