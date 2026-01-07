{{ config(enabled=var('amazon_selling_partner__using_orders_module', true)) }}

with orders as (

    select *
    from {{ ref('stg_amazon_selling_partner__orders') }}
),

order_item as (

    select *
    from {{ ref('stg_amazon_selling_partner__order_item') }}
),

{% if var('amazon_selling_partner__using_catalog_module', true) %}
item as (

    select *
    from {{ ref('int_amazon_selling_partner__item') }}
),
{% endif %}

order_item_promotion_id as (
    
    select *
    from {{ ref('stg_amazon_selling_partner__order_item_promotion_id') }}
),

aggregate_promotions as (

    select 
        source_relation,
        amazon_order_id,
        order_item_id,
        count(distinct promotion_id) as count_promotions_used

    from order_item_promotion_id 
    group by 1,2,3
),

joined as (

    select 
        order_item.*,
        orders.purchase_date as order_purchase_date,
        orders.order_total_amount, 
        orders.order_total_currency_code,
        orders.order_status,
        orders.marketplace_id,
        orders.number_of_items_shipped as order_total_number_of_items_shipped,
        orders.number_of_items_unshipped as order_total_number_of_items_unshipped,
        aggregate_promotions.count_promotions_used

        {% if var('amazon_selling_partner__using_catalog_module', true) %}
        , item.item_name,
        item.display_name,
        item.brand,
        item.color,
        item.size,
        item.style,
        item.product_type,
        item.package_quantity,
        item.item_classification
        {% endif %}

    from order_item 
    {% if var('amazon_selling_partner__using_catalog_module', true) %}
    left join item 
        on order_item.asin = item.asin 
        and order_item.source_relation = item.source_relation
    {% endif %}
    left join orders 
        on order_item.amazon_order_id = orders.amazon_order_id
        and order_item.source_relation = orders.source_relation
    left join aggregate_promotions
        on order_item.amazon_order_id = aggregate_promotions.amazon_order_id
        and order_item.order_item_id = aggregate_promotions.order_item_id
        and order_item.source_relation = aggregate_promotions.source_relation
)

select *
from joined