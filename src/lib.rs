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





impl <T: Default + for<'a>AddAssign<&'a T>> Tensor<T> {

    fn get_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;

        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride = stride * shape[i];
        }

        strides
    }


    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, TensorError> {

        if data.len() != shape.iter().product() {return Err(TensorError::InvalidShape)}
        
        let strides: Vec<usize> = Self::get_strides(&shape);

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

        let strides: Vec<usize> = Self::get_strides(&shape);

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


    pub fn transpose(&self) -> TensorView<'_, T> {
        let mut shape = self.shape.clone();
        shape.reverse();

        let mut strides = self.strides.clone();
        strides.reverse();

        TensorView {
            data: self.data.as_deref().unwrap(),
            shape,
            strides
        }
    }


    pub fn reshape(&self, shape: Vec<usize>) -> Result<TensorView<'_, T>, TensorError> {
        
        if shape.iter().product::<usize>() != self.shape.iter().product() {return Err(TensorError::InvalidShape)}

        let strides = Self::get_strides(&shape);

        Ok(TensorView {
            data: self.data.as_deref().unwrap(),
            shape,
            strides
        })
    }


    //Sum over an axis into another tensor
    //TODO: Accomodate permutations
    pub fn sum<'a>(&self, targ: &'a mut Tensor<T>, axis: usize) -> Result<TensorView<'a, T>, TensorError> {
        if axis >= self.shape.len() {return Err(TensorError::InvalidAxis)}

        let mut new_shape: Vec<usize> = self.shape.clone();

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

        let data = self.data.as_deref().unwrap();

        for i in 0..data.len() {
            let j = (i / pa) * pb + i % pb;
            targ.data.as_mut().unwrap()[j] += &data[i];
        }
        

        let new_strides = Self::get_strides(&new_shape);


        Ok(TensorView{
            data: targ.data.as_deref().unwrap(),
            shape: new_shape,
            strides: new_strides
        })
    }
}



