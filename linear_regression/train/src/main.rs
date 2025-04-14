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

use ndarray::{Array1, Array2};
use linear_regression::{LinearRegression, read_csv};
use plotters::prelude::*;
use std::error::Error;

fn plot_data(
	y_data: &Vec<f64>, x_data: &Vec<f64>,
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

	chart.configure_mesh()
		.x_desc(x_label)
		.y_desc(data_label)
		.draw()?;

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

fn find_best_lr(
	y: &Array1<f64>, x: &Array2<f64>, start_lr: f64,
	iter: usize, epoch: usize
) -> f64{
	let mut best_lr = start_lr;
	let mut lr = start_lr;
	let mut best_loss = f64::INFINITY;
	for i in 0..iter {
		let mut model = LinearRegression::new(lr, epoch);
		let loss = model.fit(x, y);
		lr /= 10.0;
		if loss.iter().any(|&x| x.is_nan()) == true {
			continue ;
		}
		if i % 10 == 0 {
			let title = format!("Loss with lr {}", lr);
			let _ = visualize_data(&loss, "Loss Value", "Generation", title.as_str());
		}
		let ave = loss.iter().sum::<f64>() / loss.len() as f64;
		if ave < best_loss {
			best_loss = ave;
			best_lr = lr;
		}
	}
	best_lr
}

fn main() {
	let res = read_csv("../data.csv");
	let Ok((header, y, x)) = res else { eprintln!("Error {}", res.unwrap_err()); return; };
	// println!("{:?}, {:?}", header[0], header[1]);
	// println!("{}", y.sum() / y.len() as f64);
	let epoch = 1e3 as usize;
	let lr = 1e-6;//find_best_lr(&y, &x, 1.0, 20, epoch);
	let mut model = LinearRegression::new(lr, epoch);
	// println!("debug y {:?}\nx {:?}", y, x);
	model.fit(&x, &y);
	println!("best lr {}, epoch {}", lr, epoch);
	model.print_out_coefficients();
	// let x_vec = x.to_vec();
	match model.fetch_coefficients() {
		Some((coeff, intercept)) => {
			let (y_vec, _offset) = y.clone().into_raw_vec_and_offset();
			let (x_vec, _offset) = x.clone().into_raw_vec_and_offset();
			let _ = plot_data(
				&y_vec, &x_vec, &header[0], &header[1],
				"Mileage against price", coeff[0], intercept
			);
		}
		None => { }
	}
	let (y_vec, _offset) = y.clone().into_raw_vec_and_offset();
	let (x_vec, _offset) = x.clone().into_raw_vec_and_offset();
	let _ = plot_data(
		&y_vec, &x_vec, &header[0], &header[1],
		"test", -34.1729662, 317443.77648
	);
}
