{{ config(enabled=var('amazon_selling_partner__using_catalog_module', true)) }}
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case

{{
    amazon_selling_partner.union_amazon_selling_partner_connections(
        connection_dictionary=var('amazon_selling_partner_sources'), 
        single_source_name='amazon_selling_partner', 
        single_table_name='item_relationship'
    )
}}