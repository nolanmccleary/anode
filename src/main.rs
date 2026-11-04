// Test executable. This links against the `anode` library crate and is where
// you'll exercise the library by hand. Reach into the lib via `anode::`.

fn main() {

    use anode::tensor_lib::*;
    

    let mut tensor1: Tensor<f32> = Tensor::from_vec(vec![0.0, 1.4, 3.5, 6.2], vec![2, 2]).unwrap();
    let mut tensor2: Tensor<f32> = Tensor::init(vec![2, 2]).unwrap();
    
    let tv1 = sum(&mut tensor1, &mut tensor2, 1);

    // println!("{}", tv1);
    // println!("{}", tensor2);
    




    
    println!("{}", anode::request_tequila());
}
