{{ config(enabled=var('amazon_selling_partner__using_orders_module', true)) }}

with orders as (

    select *
    from {{ ref('stg_amazon_selling_partner__orders')}}
),

order_item as (

    select *
    from {{ ref('stg_amazon_selling_partner__order_item') }}
),

order_item_promotion_id as (
    
    select *
    from {{ ref('stg_amazon_selling_partner__order_item_promotion_id') }}
),

payment_method_detail_item as (

    select *
    from {{ ref('stg_amazon_selling_partner__payment_method_detail_item') }}
),

aggregate_order_items as (

    select 
        source_relation,
        amazon_order_id,
        count(order_item_id) as count_order_items,
        sum(coalesce(item_price_amount, 0)) as total_item_price_amount,
        {{ fivetran_utils.string_agg('distinct item_price_currency_code', "', '") }} as item_price_currency_code,
        sum(coalesce(item_tax_amount, 0)) as total_item_tax_amount,
        {{ fivetran_utils.string_agg('distinct item_tax_currency_code', "', '") }} as item_tax_currency_code,
        sum(coalesce(shipping_discount_amount, 0)) as total_shipping_discount_amount,
        {{ fivetran_utils.string_agg('distinct shipping_discount_currency_code', "', '") }} as shipping_discount_currency_code,
        sum(coalesce(shipping_discount_tax_amount, 0)) as total_shipping_discount_tax_amount,
        {{ fivetran_utils.string_agg('distinct shipping_discount_tax_currency_code', "', '") }} as shipping_discount_tax_currency_code,
        sum(coalesce(shipping_price_amount, 0)) as total_shipping_price_amount,
        {{ fivetran_utils.string_agg('distinct shipping_price_currency_code', "', '") }} as shipping_price_currency_code,
        sum(coalesce(shipping_tax_amount, 0)) as total_shipping_tax_amount,
        {{ fivetran_utils.string_agg('distinct shipping_tax_currency_code', "', '") }} as shipping_tax_currency_code,
        sum(coalesce(promotion_discount_amount, 0)) as total_promotion_discount_amount,
        {{ fivetran_utils.string_agg('distinct promotion_discount_currency_code', "', '") }} as promotion_discount_currency_code,
        sum(coalesce(promotion_discount_tax_amount, 0)) as total_promotion_discount_tax_amount,
        {{ fivetran_utils.string_agg('distinct promotion_discount_tax_currency_code', "', '") }} as promotion_discount_tax_currency_code

    from order_item
    group by 1,2
),

aggregate_promotions as (

    select 
        source_relation,
        amazon_order_id,
        count(distinct promotion_id) as count_promotions_used

    from order_item_promotion_id 
    group by 1,2
),

aggregate_payment_methods as (

    select
        source_relation,
        amazon_order_id,
        {{ fivetran_utils.string_agg('distinct method', "', '") }} as methods

    from payment_method_detail_item
    group by 1,2
),

joined as (

    select 
        orders.*,
        aggregate_payment_methods.methods,
        aggregate_order_items.count_order_items,
        aggregate_order_items.total_item_price_amount,
        aggregate_order_items.item_price_currency_code,
        aggregate_order_items.total_item_tax_amount,
        aggregate_order_items.item_tax_currency_code,
        aggregate_order_items.total_shipping_discount_amount,
        aggregate_order_items.shipping_discount_currency_code,
        aggregate_order_items.total_shipping_discount_tax_amount,
        aggregate_order_items.shipping_discount_tax_currency_code,
        aggregate_order_items.total_shipping_price_amount,
        aggregate_order_items.shipping_price_currency_code,
        aggregate_order_items.total_shipping_tax_amount,
        aggregate_order_items.shipping_tax_currency_code,
        aggregate_order_items.total_promotion_discount_amount,
        aggregate_order_items.promotion_discount_currency_code,
        aggregate_order_items.total_promotion_discount_tax_amount,
        aggregate_order_items.promotion_discount_tax_currency_code,
        aggregate_promotions.count_promotions_used

    from orders 
    left join aggregate_order_items
        on orders.amazon_order_id = aggregate_order_items.amazon_order_id
        and orders.source_relation = aggregate_order_items.source_relation
    left join aggregate_promotions
        on orders.amazon_order_id = aggregate_promotions.amazon_order_id
        and orders.source_relation = aggregate_promotions.source_relation
    left join aggregate_payment_methods
        on orders.amazon_order_id = aggregate_payment_methods.amazon_order_id
        and orders.source_relation = aggregate_payment_methods.source_relation
)

select *
from joined