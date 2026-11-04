use std::{default, ops::AddAssign, usize};


pub fn request_tequila() -> &'static str {
    "Hola! Dos tequilas por favor"
}


pub struct Tensor <T> 
{
    data: Option<Vec<T>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}


pub struct TensorView<'a, T> 
{
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
}


#[derive(Debug)]
pub enum TensorError 
{
    InvalidShape,
    InvalidAxis,
    NoTensors,
}


fn get_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;

    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride = stride * shape[i];
    }

    strides
}



pub trait TensorInterface <T> {
    fn data(&self) -> &[T];
    fn shape(&self) -> &[usize];
    fn strides(&self) -> &[usize];
}



impl <T> TensorInterface <T> for Tensor<T> {
   fn data(&self) -> &[T] {
       self.data.as_deref().unwrap()
   } 

   fn shape(&self) -> &[usize] {
       self.shape.as_ref()
   }

   fn strides(&self) -> &[usize] {
       self.strides.as_ref()
   }
}



impl <'a, T> TensorInterface <T> for TensorView<'a, T> {
   fn data(&self) -> &'a [T] {
       self.data
   } 

   fn shape(&self) -> &[usize] {
       self.shape.as_ref()
   }

   fn strides(&self) -> &[usize] {
       self.strides.as_ref()
   }
}




impl <T: Default> Tensor<T> {

    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {

        if data.len() != shape.iter().product() {return Err(TensorError::InvalidShape)}
        
        let strides: Vec<usize> = get_strides(&shape);

        Ok(Self{
            data: Some(data),
            shape, 
            strides,
        })
    }


    pub fn init(shape: Vec<usize>) -> Result<Self, TensorError>
    where 
        T: Default,
    {
        let data = std::iter::repeat_with(T::default).take(shape.iter().product()).collect::<Vec<T>>();

        let strides: Vec<usize> = get_strides(&shape);

        Ok(Self{
            data: Some(data),
            shape,
            strides,
        })
    }

    
    pub fn view(&self) -> TensorView<'_, T> { 
        TensorView {
            data: self.data.as_deref().unwrap(),
            shape: self.shape.clone(),
            strides: self.strides.clone()
        }
    }
}


pub fn transpose<'a, S, T>(source: &'a S) -> TensorView<'a, T> 
where
    S: TensorInterface<T>,
{
    let mut shape = source.shape().to_vec();
    shape.reverse();

    let mut strides = source.strides().to_vec();
    strides.reverse();

    TensorView {
        data: source.data(), 
        shape,
        strides
    }
}


pub fn reshape<'a, S, T>(source: &'a S, shape: Vec<usize>) -> Result<TensorView<'a, T>, TensorError>
where 
    S: TensorInterface<T>
{
    
    if shape.iter().product::<usize>() != source.shape().to_vec().iter().product() {return Err(TensorError::InvalidShape)}

    let strides = get_strides(&shape);

    Ok(TensorView {
        data: source.data(), 
        shape,
        strides
    })
}



//Sum over an axis into another tensor
//TODO: Accomodate permutations
pub fn sum<'a, 'b, T, S>(source: &'a S, targ: &'b mut Tensor<T>, axis: usize) -> Result<TensorView<'b, T>, TensorError> 
where 
    S: TensorInterface<T>,
    T: Default + for<'c>AddAssign<&'c T>
{
    if axis >= source.shape().len() {return Err(TensorError::InvalidAxis)}

    let mut new_shape = source.shape().to_vec();

    let pb: usize = new_shape[axis + 1..].iter().product();
    let pa = pb * new_shape[axis];

    new_shape.remove(axis);
    let new_size = new_shape.iter().product();

    match targ.data.as_deref() {
        None => {
            targ.data = Some(std::iter::repeat_with(T::default).take(new_size).collect()); 
        },

        Some(vect) => {
            if vect.len() != new_size {return Err(TensorError::InvalidShape)}
        }
    };

    let data = source.data();

    for i in 0..data.len() {
        let j = (i / pa) * pb + i % pb;
        targ.data.as_mut().unwrap()[j] += &data[i];
    }
    

    let new_strides = get_strides(&new_shape);


    Ok(TensorView{
        data: targ.data.as_deref().unwrap(),
        shape: new_shape,
        strides: new_strides
    })
}
