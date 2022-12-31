extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(EnumFrom)]
pub fn derive_enum_from(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    enum_from_derive_impl(&input)
}

fn enum_from_derive_impl(input: &DeriveInput) -> TokenStream {
    use syn::Data::*;
    let e = match &input.data {
        Enum(e) => e,
        Struct(_) | Union(_) => panic!("#[derive(EnumFrom)] only works with enums"),
    };

    let ty = &input.ident;

    let vals = e.variants.iter().map(|v| match v.fields {
        syn::Fields::Unit => quote! { #v },
        _ => panic!("#[derive(EnumFrom)] only supports unit enums for now"),
    });

    let qs = vals
        .enumerate()
        .map(|(i, v)| {
            let i = i as u8;
            quote! { #i => Some(#ty::#v) }
        })
        .collect::<Vec<_>>();

    TokenStream::from(quote! {
        impl ::derive_enum::EnumFrom for #ty {

            fn from_u8(v: u8) -> Option<Self> {
                match v {
                    #(#qs,)*
                    _ => None,
                }
            }
        }
    })
}

#[proc_macro_derive(EnumCount)]
pub fn derive_enum_count(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    enum_count_derive_impl(&input)
}

fn enum_count_derive_impl(input: &DeriveInput) -> TokenStream {
    use syn::Data::*;
    let e = match &input.data {
        Enum(e) => e,
        Struct(_) | Union(_) => panic!("#[derive(EnumCount)] only works with enums"),
    };

    let ty = &input.ident;

    e.variants.iter().for_each(|v| match v.fields {
        syn::Fields::Unit => {}
        _ => panic!("#[derive(EnumCount)] only supports unit enums for now"),
    });

    let len = e.variants.len();

    TokenStream::from(quote! {
        impl ::derive_enum::EnumCount for #ty {
            const COUNT: usize = #len;
        }
    })
}
