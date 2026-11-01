use std::usize;


pub fn request_tequila() -> &'static str {
    "Hola! Dos tequilas por favor"
}


pub struct Tensor <T> 
{
    data: Vec<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}


pub struct TensorView<'a, T> 
{
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<usize>,
}


pub enum ShapeMismatchError 
{
    InvalidShape,
}


impl <T> Tensor<T> {

    fn get_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;

        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride = stride * shape[i];
        }

        strides
    }


    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Result<Self, ShapeMismatchError> {

        if data.len() != shape.iter().product() {return Err(ShapeMismatchError::InvalidShape)}
        
        let strides: Vec<usize> = Self::get_strides(&shape);

        Ok(Self{
            data,
            shape, 
            strides,
        })
    }


    pub fn init(shape: Vec<usize>) -> Result<Self, ShapeMismatchError>
    where 
        T: Default,
    {
        let data = std::iter::repeat_with(T::default).take(shape.iter().product()).collect::<Vec<T>>();

        let strides: Vec<usize> = Self::get_strides(&shape);

        Ok(Self{
            data,
            shape,
            strides,
        })
    }

    
    pub fn view(&self) -> TensorView<'_, T> { 
        TensorView {
            data: &self.data,
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
            data: &self.data,
            shape,
            strides
        }
    }


    pub fn reshape(&self, shape: Vec<usize>) -> Result<TensorView<'_, T>, ShapeMismatchError> {
        
        if shape.iter().product::<usize>() != self.shape.iter().product() {return Err(ShapeMismatchError::InvalidShape)}

        let strides = Self::get_strides(&shape);

        Ok(TensorView {
            data: &self.data,
            shape,
            strides
        })
    }
}



