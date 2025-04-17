/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lib.rs                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: jngerng <jngerng@student.42kl.edu.my>      +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/17 17:36:07 by jngerng           #+#    #+#             */
/*   Updated: 2025/04/17 17:42:12 by jngerng          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

use ndarray::Array1;
use plotters::prelude::*;
use std::error::Error;

fn find_scale_factor(min: f64) -> (f64, i32) {
	let order_of_magnitude = min.log10().floor() as i32;
	(10f64.powi(order_of_magnitude), order_of_magnitude)
}

pub fn plot_data(
	y_data: &Array1<f64>, x_data: &Array1<f64>,
	data_label: &str, x_label: &str, title: &str,
	coefficient: f64, intercept: f64
) -> Result<(), Box<dyn Error>> {
	let output_file = format!("{}.jpeg", title);
	let root = BitMapBackend::new(&output_file, (1000, 800)).into_drawing_area();
	// let label = format!("{} Over Generations", data_label);
	root.fill(&WHITE)?;
	let x_min = x_data.iter().cloned().fold(f64::INFINITY, f64::min);
	let x_max = x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let y_min = y_data.iter().cloned().fold(f64::INFINITY, f64::min);
	let y_max = y_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let (x_scale, x_factor) = find_scale_factor(x_min);
	let (y_scale, y_factor) = find_scale_factor(y_min);

	// Create an iterator with a custom step for f64 values
	let step = (x_max - x_min) / 100.0; // Adjust the number of points
	let x_values = (0..=100)
		.map(|i| x_min + step * i as f64)  
		.into_iter();

	let x_step = (x_max - x_min) / x_data.len() as f64;
	let y_step = (x_max - x_min) / x_data.len() as f64;
	let x_range = (x_min - x_step)..(x_max + x_step);
	let y_range = (y_min - y_step)..(y_max + y_step);

	let mut chart = ChartBuilder::on(&root)
		.caption(title, ("sans-serif", 30))
		.margin(30)
		.x_label_area_size(30)
		.y_label_area_size(50)
		.build_cartesian_2d(x_range, y_range)?;

	chart.configure_mesh()
		.x_desc(format!("{}, 1e{}", x_label, x_factor))
		.y_desc(format!("{}, 1e{}", data_label, y_factor))
		.x_labels(x_data.len())
		.y_labels(y_data.len())
		.x_label_formatter(&|x| format!("{:.1}", x / x_scale))
        .y_label_formatter(&|y| format!("{:.1}", y / y_scale))
		.draw()?;

	// Draw scatter plot (points)
	chart.draw_series(
		x_data.iter()
			.cloned()
			.zip(y_data.iter().cloned())
			.map(|(x, y)| Circle::new((x, y), 5, BLUE.filled())),
	)?
	.label("Data Points")
	.legend(|(x, y)| Circle::new((x + 10, y), 5, BLUE.filled()));

	// Draw the line y = ax + b (from x = 0 to x = 6)
	chart.draw_series(LineSeries::new(
		x_values.map(|x| (x, coefficient * x + intercept)), // Line points (x, y)
		&RED,
	))?
	.label(format!("{:.1}x + {:.1}", coefficient, intercept))
	.legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));
	
	chart.configure_series_labels()
		.border_style(&BLACK)
		.draw()?;
	println!("Plot saved to {}", output_file);
	Ok(())
}


pub fn visualize_data(
	data: &Vec<f64>, data_label: &str, x_label: &str, title: &str
) -> Result<(), Box<dyn Error>> {
	let output_file = format!("{}.jpeg", title);
	let root = BitMapBackend::new(&output_file, (1000, 800)).into_drawing_area();
	// let label = format!("{} Over Generations", data_label);
	root.fill(&WHITE)?;
	let y_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
	let y_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
	let (y_scale, y_factor) = find_scale_factor(y_min);

	let mut chart = ChartBuilder::on(&root)
		.caption(title, ("sans-serif", 30))
		.margin(30)
		.x_label_area_size(30)
		.y_label_area_size(50)
		.build_cartesian_2d(0..data.len(), y_min..y_max)?;

	chart.configure_mesh()
		.x_desc(x_label)
		.y_desc(format!("{}, 1e{}", data_label, y_factor))
		.y_label_formatter(&|y| format!("{:.1}", y / y_scale))
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
		
		// lazy to test lulz
	}
}
