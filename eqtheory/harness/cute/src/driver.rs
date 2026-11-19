use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

fn main() {
    let sock = std::env::args().nth(1).expect("usage: cutedriver <socket>");
    let mut s = UnixStream::connect(&sock).expect("connect");
    s.write_all(&[1u8]).unwrap();
    let mut len = [0u8; 4];
    s.read_exact(&mut len).unwrap();
    let n = u32::from_le_bytes(len) as usize;
    let mut buf = vec![0u8; n];
    s.read_exact(&mut buf).unwrap();
    print!("{}", String::from_utf8(buf).unwrap());
}
