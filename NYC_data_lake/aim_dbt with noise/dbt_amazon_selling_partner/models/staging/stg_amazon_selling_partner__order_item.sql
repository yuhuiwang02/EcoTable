{{ config(enabled=var('amazon_selling_partner__using_orders_module', true)) }}

with base as (

    select * 
    from {{ ref('stg_amazon_selling_partner__order_item_base') }}
),

fields as (

    select
        {{
            fivetran_utils.fill_staging_columns(
                source_columns=adapter.get_columns_in_relation(ref('stg_amazon_selling_partner__order_item_base')),
                staging_columns=get_order_item_columns()
            )
        }}
        
        {{ amazon_selling_partner_apply_source_relation() }}
        
    from base
),

final as (
    
    select 
        source_relation, 
        amazon_order_id,
        order_item_id,
        cast(asin as {{ dbt.type_string() }}) as asin,
        seller_sku,
        title,
        product_info_detail_number_of_items,
        scheduled_delivery_start_date,
        scheduled_delivery_end_date,
        quantity_ordered,
        quantity_shipped,
        {{ amazon_selling_partner.convert_string_to_numeric('item_price_amount') }} as item_price_amount,
        item_price_currency_code,
        {{ amazon_selling_partner.convert_string_to_numeric('item_tax_amount') }} as item_tax_amount,
        item_tax_currency_code,
        {{ amazon_selling_partner.convert_string_to_numeric('shipping_discount_amount') }} as shipping_discount_amount,
        shipping_discount_currency_code,
        {{ amazon_selling_partner.convert_string_to_numeric('shipping_discount_tax_amount') }} as shipping_discount_tax_amount,
        shipping_discount_tax_currency_code,
        {{ amazon_selling_partner.convert_string_to_numeric('shipping_price_amount') }} as shipping_price_amount,
        shipping_price_currency_code,
        {{ amazon_selling_partner.convert_string_to_numeric('shipping_tax_amount') }} as shipping_tax_amount,
        shipping_tax_currency_code,
        {{ amazon_selling_partner.convert_string_to_numeric('promotion_discount_amount') }} as promotion_discount_amount,
        promotion_discount_currency_code,
        {{ amazon_selling_partner.convert_string_to_numeric('promotion_discount_tax_amount') }} as promotion_discount_tax_amount,
        promotion_discount_tax_currency_code,
        condition_id,
        condition_note,
        condition_subtype_id,
        buyer_requested_cancel_buyer_cancel_reason as buyer_requested_cancel_reason,
        buyer_requested_cancel_is_buyer_requested_cancel as is_buyer_requested_cancel,
        deemed_reseller_category,
        ioss_number,
        is_gift,
        is_transparency,
        serial_number_required as is_serial_number_required, -- only populated for Easy Ship orders
        store_chain_store_id,
        tax_collection_model, -- always MarketplaceFacilitator in US
        tax_collection_responsible_party -- always Amazon Web Services in US
        
    from fields
)

select *
from final
