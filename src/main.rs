// Test executable. This links against the `anode` library crate and is where
// you'll exercise the library by hand. Reach into the lib via `anode::`.

fn main() {




    

    let tensor1: anode::Tensor<f32> = anode::Tensor::from_vec(vec![0.0, 1.4, 3.5, 6.2], vec![2, 2]).unwrap();
    let mut tensor2: anode::Tensor<f32> = anode::Tensor::init(vec![2, 2]).unwrap();
    
    let tv1 = tensor1.sum(&mut tensor2, 1);

    // println!("{}", tv1);
    // println!("{}", tensor2);
    




    
    println!("{}", anode::request_tequila());
}
