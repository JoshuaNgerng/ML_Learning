/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.rs                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: jngerng <jngerng@student.42kl.edu.my>      +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/26 14:04:50 by jngerng           #+#    #+#             */
/*   Updated: 2025/04/17 17:47:53 by jngerng          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

use linear_regression::{LinearRegression, read_csv};
use visualize::{visualize_data, plot_data};
use std::error::Error;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn Error>> {
	let (header, y, x) = read_csv("../data.csv")?;
	let epoch = 100;
	let lr = 1e-1;
	let mut model = LinearRegression::new(lr, epoch);
	let loss = model.fit(&x, &y, true);
	if loss.iter().any(|&x| x.is_nan()) == false {
		visualize_data(&loss, "Loss Value", "Iteration", "Loss Over Iteration")?;
	}
	model.print_out_coefficients();
	match model.fetch_coefficients() {
		Some((coeff, intercept)) => {
			let flatten_x = x.to_shape(x.len()).unwrap().to_owned();
			if coeff[0].is_nan() || intercept.is_nan() {
				eprintln!("No Coefficient converge to, try having a smaller learning rate");
				return Err("No Coefficients for model found".into());
			}
			let _ = plot_data(
				&y, &flatten_x, &header[0], &header[1],
				"Mileage against price", coeff[0], intercept
			);
			let mut file = File::create("../result.txt")?;
			let text = format!("{} {}\n", coeff[0], intercept);
			file.write_all(text.as_bytes())?;
		}
		None => { println!("No Coefficient converge to, try having a smaller learning rate"); }
	}
	Ok(())
}
