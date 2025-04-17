/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.rs                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: jngerng <jngerng@student.42kl.edu.my>      +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/17 14:58:09 by jngerng           #+#    #+#             */
/*   Updated: 2025/04/17 17:16:08 by jngerng          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

use regex::Regex;
use std::{env, f64};
use std::error::Error;
use std::fs::File;
use std::io::{stdin, stdout, Write, Read};

fn get_name() -> Result<String, Box<dyn Error>> {
	let arg = env::args().collect::<Vec<String>>();
	if arg.len() != 2 {
		return Err("Please give fname for coefficients as argument".into());
	}
	Ok(arg[1].to_owned())
}

fn main() -> Result<(), Box<dyn Error>>{
	let fname = get_name()?;
	let mut file = File::open(fname)?;
	let mut text = String::new();
	file.read_to_string(&mut text)?;
	let re = Regex::new(r"([+-]?\d+\.\d+)")?;

	let cofficients: Vec<f64> = re.find_iter(&text)
		.map(|mat| mat.as_str().parse::<f64>().unwrap())  // Convert matches to Strings
		.collect();
	
	if cofficients.len() != 2 {
		return Err("Wrong number of coefficients in input file".into());
	}

	let mut input = String::new();
	print!("Enter price: ");
	stdout().flush()?;
	stdin().read_line(&mut input)?;

    let input: f64 = input.trim().parse().map_err(|e| {
        // If parsing fails, return a custom error wrapped in Box<dyn Error>
        Box::new(e) as Box<dyn Error>
    })?;
	
	let res = cofficients[0] * input + cofficients[1];

	println!("predicted km is {}", res);

	Ok(())
}

/*
Regex Pattern r"([+-]?\d+\.\d+)": This regex will match floating-point numbers. It works as follows:

    [+-]?: Matches an optional sign (+ or -).

    \d+: Matches one or more digits before the decimal point (integer part).

    \.: Matches the decimal point ..

    \d+: Matches one or more digits after the decimal point (fractional part).
*/