{{ config(
    tags="fivetran_validations",
    enabled=var('fivetran_validation_tests_enabled', false)
) }}

with source as (

    select 
        count(*) as row_count,
        count(distinct amazon_order_id) as count_distinct_orders,
        round(sum(coalesce(order_total_amount, 0)), 2) as order_total_amount,
        sum(coalesce(number_of_items_shipped, 0)) as number_of_items_shipped,
        sum(coalesce(number_of_items_unshipped, 0)) as number_of_items_unshipped
    from {{ target.schema }}_amazon_selling_partner_dev.stg_amazon_selling_partner__orders
),

model as (

    select 
        count(*) as row_count,
        count(distinct amazon_order_id) as count_distinct_orders,
        round(sum(coalesce(order_total_amount, 0)), 2) as order_total_amount,
        sum(coalesce(number_of_items_shipped, 0)) as number_of_items_shipped,
        sum(coalesce(number_of_items_unshipped, 0)) as number_of_items_unshipped
    from {{ target.schema }}_amazon_selling_partner_dev.amazon_selling_partner__orders
)

select 
    model.row_count as model_row_count,
    source.row_count as source_row_count,
    model.count_distinct_orders as model_count_distinct_orders,
    source.count_distinct_orders as source_count_distinct_orders,
    model.order_total_amount as model_order_total_amount,
    source.order_total_amount as source_order_total_amount,
    model.number_of_items_shipped as model_number_of_items_shipped,
    source.number_of_items_shipped as source_number_of_items_shipped,
    model.number_of_items_unshipped as model_number_of_items_unshipped,
    source.number_of_items_unshipped as source_number_of_items_unshipped

from model 
join source on true
where 
    model.row_count != source.row_count or
    model.count_distinct_orders != source.count_distinct_orders or
    model.order_total_amount != source.order_total_amount or
    model.number_of_items_shipped != source.number_of_items_shipped or
    model.number_of_items_unshipped != source.number_of_items_unshipped