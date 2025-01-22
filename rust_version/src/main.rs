use std::{env, process};
use string_oscillations::{make_cardioid, Settings};

fn main() {
    let settings = Settings::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments: {}", err);
        process::exit(1);
    });

    let string_order = make_cardioid(&settings);
    println!("Computed {} strings.", string_order.len())
}
