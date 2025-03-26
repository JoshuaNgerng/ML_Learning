/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   main.rs                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: jngerng <jngerng@student.42kl.edu.my>      +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/03/26 14:04:50 by jngerng           #+#    #+#             */
/*   Updated: 2025/03/26 16:03:17 by jngerng          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

use linear_regression::{ read_csv};

fn main() {
    let res = read_csv("../data.csv");
    let Ok((header, y, x)) = res else { eprintln!("Error {}", res.unwrap_err()); return; };
    println!("{:?}", header);
    println!("{:?}", y);
    println!("{:?}", x);
}
