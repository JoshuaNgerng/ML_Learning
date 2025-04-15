/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lib.rs                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: jngerng <jngerng@student.42kl.edu.my>      +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/26 10:34:17 by jngerng           #+#    #+#             */
/*   Updated: 2025/04/15 01:11:23 by jngerng          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

use ndarray::{Array1, Array2, concatenate, Axis, s};
use csv::ReaderBuilder;
use std::error::Error;
use std::fmt;

pub struct LinearRegression {
	coefficients: Option<Array1<f64>>,
	constant: f64,
	learning_rate: f64,
	epochs: usize
}

/*
revise my algo
https://www.youtube.com/watch?v=O96hzKRx3O4
https://stackoverflow.com/questions/42869949/gradient-descent-on-linear-regression-not-converging
*/

impl LinearRegression {
	pub fn new(learning_rate: f64, epochs: usize) -> Self {
		LinearRegression {
			coefficients: None, constant: 0.0, learning_rate, epochs
		}
	}
	pub fn fit(
		&mut self, x: &Array2<f64>, y: &Array1<f64>,
	) -> Vec<f64> {
		let observ = x.nrows();
		let feature = x.ncols();
		// println!("debug obs{}, fea{}", observ, feature);
		if y.len() != observ {
			panic!("The number of observations in y must match the number of rows in x.")
		}
		let feature_matrix = self.normalize_n_pad_feature_matrix(x);
		// println!("{:?}", feature_matrix);
		let mut coeff = Array1::<f64>::zeros(feature + 1);
		coeff[0] = y.sum() / y.len() as f64;
		let mut lost_history = Vec::<f64>::new();
		// println!("debug {:?} {:?}", coeff, x.t());
		
		for i in 0..self.epochs {
			let predict = feature_matrix.dot(&coeff);
			let residual = &predict - y;
			// if i % 1000 == 0 {
				// println!("{}\n{}", predict, y);
			// }
			// println!("debug residual {:?}", residual);
			let gradient = (2.0 / observ as f64) * (feature_matrix.t().dot(&residual));
			let loss = self.cal_loss(&residual);
			// println!("debug grad {:?} {}", gradient, shift);
			coeff = &coeff - (self.learning_rate * &gradient);
			lost_history.push(loss);
			if i % 1000 == 0 {
				// println!("{}\n", residual);
				println!("{}", i);
			}
		}
		let slice_view = coeff.slice(s![1..; 1]);
		self.coefficients = Some(slice_view.to_owned());
		self.constant = coeff[0];
		lost_history
	}
	pub fn predict(&self, x: &Array2<f64>) -> Option<Array1<f64>> {
		match &self.coefficients {
			Some(coeff) => {
				if coeff.len() != x.nrows() {
					return None
				}
				Some(x.dot(coeff))
			},
			_ => None,
		}
	}
	pub fn assign_coeffcients(&mut self, coefficient: &Array1<f64>) {
		self.coefficients = Some(coefficient.clone());
	}
	pub fn print_out_coefficients(&self) {
		if let Some(coeff) = &self.coefficients {
			println!("{} {}", coeff, self.constant);
		} else {
			println!("None");
		}
	}
	fn cal_loss(&self, residual: &Array1<f64>) -> f64 {
		let res = residual * residual;
		res.sum() / res.len() as f64
	}
	pub fn fetch_coefficients(&self) -> Option<(Array1<f64>, f64)> {
		if let Some(coeff) = &self.coefficients {
			return Some((coeff.clone(), self.constant));
			// println!("{} {}", coeff, self.constant);
		}
		None
	}
	fn normalize_n_pad_feature_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
		let normal_x = (x - x.mean().unwrap()) / x.std(0.0);
		let new_column = Array2::from_elem((x.nrows(), 1), 1.0);
		return concatenate![Axis(1), new_column, normal_x];
		// return concatenate![Axis(1), new_column, *x];
	}
}


#[derive(Debug)]
pub struct CsvError {
	message: String,
}

impl fmt::Display for CsvError {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", self.message)
	}
}

impl Error for CsvError {}

pub fn read_csv(
	file_path: &str
) -> Result<(Vec<String>, Array1<f64>, Array2<f64>), Box<dyn Error>> {
	let mut rdr = ReaderBuilder::new().has_headers(true).from_path(file_path)?;

	let header = rdr.headers()?.iter().map(|s| s.to_string()).collect();
	let buffer: Vec<_> = rdr.records().collect::<Result<_, _>>()?;
	let nrows = buffer.len();
	let ncols = if nrows > 0 { buffer[0].len() } else { 0 };
	let mut y = Array1::<f64>::zeros(nrows);
	let mut x = Array2::<f64>::zeros((nrows, ncols - 1));
	for (row_index, record) in buffer.iter().enumerate() {
		for (col_index, field) in record.iter().enumerate() {
			let value;
			match field.parse::<f64>() {
				Ok(check) => value = check,
				Err(_) => {
					return Err(Box::new(CsvError {
						message: format!(
							"Input file contains non-numerical data at row {}, col {}: '{}'",
							row_index, col_index, field
						),
					}));
				}
			}
			if col_index == 0 {
				y[row_index] = value;
			} else {
				x[[row_index, col_index - 1]] = value;
			}
		}
	}

	Ok((header, y, x))
}

#[cfg(test)]
mod tests {
	use super::*;
	use ndarray::arr2;

	#[test]
	fn it_fits_linear_model() {
		let mut model = LinearRegression::new(0.01, 1000);
		
		let x_data = arr2(&[
			[1.0, 1.0],
			[1.0, 2.0],
			[2.0, 2.0],
			[2.0, 3.0],
			[3.0, 3.0],
		]);
		
		let y_data = Array1::from_vec(vec![1.0, 2.0, 2.0, 3.0, 4.0]);

		model.fit(&x_data, &y_data);

		assert!(model.coefficients.is_some()); // Check that coefficients are computed
		let coefficients = model.coefficients.as_ref().unwrap();
		assert_eq!(coefficients.len(), 2); // Ensure the number of coefficients matches the number of features
	}

	#[test]
	#[should_panic(expected = "The number of observations in y must match the number of rows in x.")]
	fn it_panics_on_dimension_mismatch() {
		let mut model = LinearRegression::new(0.01, 1000);
		
		let x_data = arr2(&[
			[1.0, 1.0],
			[1.0, 2.0],
		]);
		
		let y_data = Array1::from_vec(vec![1.0]); // Only one observation

		model.fit(&x_data, &y_data); // This should panic
	}
	#[test]
	fn test_read_csv() {
		let (header, y, x) = read_csv("../data.csv").unwrap();
		assert_eq!(header.len(), 2);
		assert_eq!(header[0], "km");
		assert_eq!(header[1], "price");
		assert_eq!(y.len(), 24);
		assert_eq!(x.ncols(), 1);
		assert_eq!(x.nrows(), 24);
		assert_eq!(y[1], 139800 as f64);
		assert_eq!(x[[2, 0]], 4400 as f64);
	}
}

/*
OLS Ordinary Least Square Method (is a method for linear)
Gradient Descent is the method subject pdf wants
y = aX + e
mean square error (minize e, residual error)
e = 1/n . sum (yi - (m.xi + b)^2)
de/dm = 1/n . sum (2. (yi - (m.xi + b)) . (-xi))
= -2/n . sum ( xi (yi - (m.xi + b)) )
de/db = -2/n 0 sum ( yi - (m.xi + b))
m = m - L de /dm 
b = b - L . de / db
where L is learning rate
might have many feature for assume only two
big L faster, small L better res ex (0.001)
*/