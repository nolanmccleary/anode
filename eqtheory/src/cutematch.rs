use cute_layout::algebra::mapping::composition;
use cute_layout::Layout;
use eqtheory::cute_match::Kernel;

fn show(kernel: &Kernel, label: &str, extents: &[i64]) {
    println!("\n  {label}");
    let fits = kernel.bind(extents);
    if fits.is_empty() {
        println!("    API admits no tiling");
        return;
    }
    for f in fits {
        println!("    {:<5} waste {:>4.2}x  access {}", f.tiling, f.waste, f.access);
    }
}

fn main() {
    let matmul = Kernel {
        name: "tiled_matmul",
        rank: 2,
        tilings: vec![
            ("t16", vec![16, 16]),
            ("t32", vec![32, 32]),
            ("t64", vec![64, 64]),
        ],
    };

    println!("ADMISSIBILITY IS THE CUTE API, not my arithmetic.");
    println!("bind calls zipped_divide; Ok => tiling admits, and the access + waste fall out.");

    show(&matmul, "(1024,1024)", &[1024, 1024]);
    show(&matmul, "(48,48)", &[48, 48]);
    show(&matmul, "(8,8)  <- hand-check wrongly rejected this; API rounds it", &[8, 8]);

    println!("\nwaste is derived from the API's own rounding -- THAT is the 'too small to tile'");
    println!("signal (a cost, e.g. drop waste>2x), never a divisibility gate I reimplement.\n");

    println!("the API rejects only genuine stride conflicts coalesce cannot fix:");
    let a: Layout = "(4,6,8):(2,3,5)".parse().unwrap();
    let b: Layout = "6:3".parse().unwrap();
    match composition(&a, &b) {
        Ok(r) => println!("  (4,6,8):(2,3,5) o 6:3 -> {r}"),
        Err(e) => println!("  (4,6,8):(2,3,5) o 6:3 -> Err ({e})"),
    }
}
