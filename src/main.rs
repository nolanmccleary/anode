// Test executable. This links against the `anode` library crate and is where
// you'll exercise the library by hand. Reach into the lib via `anode::`.

fn main() {
    println!("{}", anode::request_tequila());
}
