extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(Rand)]
pub fn derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    rand_derive_impl(&input)
}

fn rand_derive_impl(input: &DeriveInput) -> TokenStream {
    use syn::Data::*;
    let e = match &input.data {
        Enum(e) => e,
        Struct(_) | Union(_) => panic!("#[derive(Rand)] only works with enums"),
    };

    let ty = &input.ident;

    let mut vals = e.variants.iter().map(|v| match v.fields {
        syn::Fields::Unit => quote! { #v },
        _ => panic!("#[derive(Rand)] only supports unit enums for now"),
    });

    let len = e.variants.len();

    let body = match len {
        0 => panic!("cannot #[derive(Rand)] on empty enum"),
        1 => {
            let a = vals.next().unwrap();
            assert!(vals.next().is_none());
            quote! { #ty::#a }
        }
        2 => {
            let a = vals.next().unwrap();
            let b = vals.next().unwrap();
            assert!(vals.next().is_none());
            quote! { if __rng.gen() { #ty::#b } else { #ty::#a } }
        }
        _ => {
            let qs = vals
                .enumerate()
                .map(|(i, v)| {
                    quote! { #i => #ty::#v }
                })
                .collect::<Vec<_>>();
            quote! { match __rng.gen_range(0..#len) {
                #(#qs,)*
                _ => unreachable!(),
            } }
        }
    };

    TokenStream::from(quote! {
        impl ::rand::distributions::Distribution<#ty> for ::rand::distributions::Standard {
            fn sample<R: ::rand::Rng + ?Sized>(&self, __rng: &mut R) -> #ty {
                #body
            }
        }
    })
}
