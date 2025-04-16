/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.rs                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: jngerng <jngerng@student.42kl.edu.my>      +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/26 14:04:50 by jngerng           #+#    #+#             */
/*   Updated: 2025/04/15 00:55:20 by jngerng          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

use ndarray::Array1;
use linear_regression::{LinearRegression, read_csv};
use plotters::prelude::*;
use std::{error::Error, f64};
use std::fs::File;
use std::io::Write;

fn plot_data(
	y_data: &Array1<f64>, x_data: &Array1<f64>,
	data_label: &str, x_label: &str, title: &str,
	coefficient: f64, intercept: f64
) -> Result<(), Box<dyn Error>> {
	let output_file = format!("{}.jpeg", title);
	let root = BitMapBackend::new(&output_file, (800, 600)).into_drawing_area();
	// let label = format!("{} Over Generations", data_label);
	root.fill(&WHITE)?;
	let x_min = x_data.iter().cloned().fold(f64::INFINITY, f64::min);
	let x_max = x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let y_min = y_data.iter().cloned().fold(f64::INFINITY, f64::min);
	let y_max = y_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

	// Create an iterator with a custom step for f64 values
	let step = (x_max - x_min) / 100.0; // Adjust the number of points
	let x_values: Vec<f64> = (0..=100)
		.map(|i| x_min + step * i as f64)  // Generate x values from x_min to x_max
		.collect();

	let mut chart = ChartBuilder::on(&root)
		.caption(title, ("sans-serif", 30))
		.margin(20)
		.x_label_area_size(40)
		.y_label_area_size(40)
		.build_cartesian_2d(x_min..x_max, y_min..y_max)?;

	chart.configure_mesh()
		.x_desc(x_label)
		.y_desc(data_label)
		.draw()?;

	// Draw scatter plot (points)
	chart.draw_series(
		x_data.iter()
			.cloned()
			.zip(y_data.iter().cloned())
			.map(|(x, y)| Circle::new((x, y), 5, BLUE.filled())),
	)?;

	// Draw the line y = ax + b (from x = 0 to x = 6)
	chart.draw_series(LineSeries::new(
		x_values.iter().map(|&x| (x, coefficient * x + intercept)), // Line points (x, y)
		&RED,
	))?;
	
	chart.configure_series_labels()
		.border_style(&BLACK)
		.draw()?;
	println!("Plot saved to {}", output_file);
	Ok(())
}


fn visualize_data(
	data: &Vec<f64>, data_label: &str, x_label: &str, title: &str
) -> Result<(), Box<dyn Error>> {
	let output_file = format!("{}.jpeg", title);
	let root = BitMapBackend::new(&output_file, (800, 600)).into_drawing_area();
	// let label = format!("{} Over Generations", data_label);
	root.fill(&WHITE)?;
	let y_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
	let y_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

	let mut chart = ChartBuilder::on(&root)
		.caption(title, ("sans-serif", 30))
		.margin(20)
		.x_label_area_size(40)
		.y_label_area_size(40)
		.build_cartesian_2d(0..data.len(), y_min..y_max)?;
	// println!("test1");

	chart.configure_mesh()
		.x_desc(x_label)
		.y_desc(data_label)
		.draw()?;

	// println!("huh");
	chart.draw_series(LineSeries::new(
		(0..data.len()).zip(data.iter().cloned()),
		&BLUE,
	))?
	.label(data_label)
	.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
	
	chart.configure_series_labels()
		.border_style(&BLACK)
		.draw()?;
	println!("Plot saved to {}", output_file);
	Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let (header, y, x) = read_csv("../data.csv")?;
	let epoch = 100;
	let lr = 1e-1;
	let mut model = LinearRegression::new(lr, epoch);
	let loss = model.fit(&x, &y);
	if loss.iter().any(|&x| x.is_nan()) == false {
		visualize_data(&loss, "Loss Value", "Iteration", "Loss Over Iteration")?;
	}
	model.print_out_coefficients();
	match model.fetch_coefficients() {
		Some((coeff, intercept)) => {
			let flatten_x = x.to_shape(x.len()).unwrap().to_owned();
			if coeff[0] == f64::NAN || intercept == f64::NAN {
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
