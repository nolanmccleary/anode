fn main() {
    let mlxlib = std::env::var("DEP_MLXSHIM_MLXLIB").unwrap();
    println!("cargo:rustc-link-arg-bin=cutedriver=-Wl,-rpath,{mlxlib}");
    println!("cargo:rustc-link-arg-cdylib=-Wl,-rpath,{mlxlib}");
}
