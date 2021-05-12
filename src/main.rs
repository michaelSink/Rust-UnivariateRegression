use std::error::Error;
use std::process;

struct UnivariantLinear{
    weight: f64,
    bias: f64,
    alpha: f64,
}

impl UnivariantLinear{
    fn new() -> UnivariantLinear{
        UnivariantLinear{
            weight: 0.0,
            bias: 0.0,
            alpha: 0.01, 
        }
    }

    fn fit(&mut self, data : Vec<(f64, f64)>, epochs : i32){

        let size = data.len() as f64;
        let mut loss : f64 = 0.0;
        let mut y_hat : Vec<f64>;
        let mut d_weight : f64 = 0.0;
        let mut d_bias : f64 = 0.0;

        for _iter in 0..epochs{

            //Predict the values based on the current weight and bias
            y_hat = data.iter().map(|x| x.0 * self.weight + self.bias).collect();

            //Calculate loss using MSE
            loss = 1.0/(2.0 * size) * data.iter().zip(y_hat.iter()).map(|(y, y_h)| (y_h - y.1).powi(2)).sum::<f64>();

            //Calculate derivatives of weights
            d_weight = data.iter().zip(y_hat.iter()).map(|(y, y_h)| (y_h - y.1) * y.0).sum::<f64>();
            d_bias = data.iter().zip(y_hat.iter()).map(|(y, y_h)| (y_h - y.1)).sum::<f64>();

            //Adjust derivatives based on size of input
            d_weight = d_weight / size;
            d_bias = d_bias / size;

            //Update weight and bias
            self.weight = self.weight - self.alpha * d_weight;
            self.bias = self.bias - self.alpha * d_bias;

            if(_iter % 100 == 0){
                println!("Loss at iteration {}: {}", _iter, loss);
            }

        }

        //Print final weight and bias
        println!("Weight: {}. Bias: {}",self.weight, self.bias);
    }
}

fn parse_csv(records : &mut Vec<(f64, f64)>) -> Result<(), Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path("C:/Machine Learning/univariant-linear-regression/src/train.csv")?;
    for result in rdr.records() {
        let row: (f64, f64) = result?.deserialize(None)?;
        records.push(row);
    }
    Ok(())
}

fn main() {
    let mut records : Vec<(f64, f64)> = Vec::new();
    if let Err(err) = parse_csv(&mut records){
        println!("Error resulting from reading CSV: {}", err);
        process::exit(1);
    }
    let mut regression_model = UnivariantLinear::new();
    regression_model.fit(records, 1000);
}