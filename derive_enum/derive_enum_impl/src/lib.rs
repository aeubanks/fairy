extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(EnumCount)]
pub fn derive(input: TokenStream) -> TokenStream {
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
