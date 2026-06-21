/// LyCORIS algorithm implementations
pub mod locon;
pub mod loha;
pub mod lokr;
pub mod full;
pub mod oft;
pub mod boft;

pub use locon::LoConModule;
pub use loha::LoHaModule;
pub use lokr::LoKrModule;
pub use full::FullAdapter;
pub use oft::OFTModule;
pub use boft::BOFTModule;
