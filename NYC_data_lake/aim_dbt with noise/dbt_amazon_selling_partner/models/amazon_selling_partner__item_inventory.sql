{{ config(enabled=var('amazon_selling_partner__using_catalog_module', true)) }}

with item as (

    select *
    from {{ ref('int_amazon_selling_partner__item') }}
),

{% if var('amazon_selling_partner__using_fba_module', true) %}
fba_inventory_summary as (

    select *
    from {{ ref('stg_amazon_selling_partner__fba_inventory_summary') }}
),

fba_inventory_researching_quantity_entry as (
    select * 
    from {{ ref('stg_amazon_selling_partner__fba_inventory_researching') }}
),

pivot_researching_quantity as (
    
    select 
        inventory_summary_id,
        source_relation,
        sum(case when lower(name) = 'researchingquantityinshortterm' then quantity else 0 end) as short_term_research_quantity,
        sum(case when lower(name) = 'researchingquantityinmidterm' then quantity else 0 end) as mid_term_research_quantity,
        sum(case when lower(name) = 'researchingquantityinlongterm' then quantity else 0 end) as long_term_research_quantity
    from fba_inventory_researching_quantity_entry
    group by 1,2
),
{% endif %}

joined as (

    select 
        -- Item description
        item.source_relation,
        item.marketplace_id,
        item.asin,
        item.item_name,
        item.display_name,
        {% if var('amazon_selling_partner__using_fba_module', true) -%} fba_inventory_summary.product_name, {%- endif %}
        item.brand,
        item.color,
        item.size,
        item.style,
        item.package_quantity,
        item.manufacturer,
        item.contributors,
        item.product_type,
        {% if var('amazon_selling_partner__using_fba_module', true) -%} fba_inventory_summary.condition, {%- endif %}
        item.release_date,
        item.item_classification,
        item.classification_id,
        item.classification_sales_rank_link,
        item.classification_sales_rank,
        item.website_display_group,
        item.website_display_group_name,
        item.website_display_group_sales_rank_link,
        item.website_display_group_sales_rank,
        item.is_memorabilia,
        item.is_adult_product,
        item.is_autographed,
        item.is_trade_in_eligible,

        -- IDs
        item.model_number,
        item.part_number,
        item.parent_variation_asin,
        item.parent_package_container_asin,
        {% if var('amazon_selling_partner__using_fba_module', true) %}
            coalesce(item.sku, fba_inventory_summary.seller_sku) as sku,
            fba_inventory_summary.fn_sku,
        {% else %}
            item.sku,
        {% endif %}
        item.ean,
        item.gtin,
        item.isbn,
        item.jan,
        item.minsan,
        item.upc,

        -- Item listing metadata 
        item.count_images,
        item.count_swatch_images,
        item.item_height_unit,
        item.item_height_value,
        item.item_length_unit,
        item.item_length_value,
        item.item_weight_unit,
        item.item_weight_value,
        item.item_width_unit,
        item.item_width_value,
        item.package_height_unit,
        item.package_height_value,
        item.package_length_unit,
        item.package_length_value,
        item.package_weight_unit,
        item.package_weight_value,
        item.package_width_unit,
        item.package_width_value

        -- Inventory description
        {% if var('amazon_selling_partner__using_fba_module', true) %}
        , fba_inventory_summary.inventory_summary_id,
        fba_inventory_summary.last_updated_at as inventory_last_updated_at,
        fba_inventory_summary.total_quantity,
        fba_inventory_summary.total_researching_quantity,
        fba_inventory_summary.total_reserved_quantity,
        fba_inventory_summary.fullfillable_quantity,
        fba_inventory_summary.total_unfulfillable_quantity,
        fba_inventory_summary.pending_customer_order_quantity,
        fba_inventory_summary.pending_transshipment_quantity,
        fba_inventory_summary.fc_processing_quantity,
        fba_inventory_summary.inblound_shipped_quantity,
        fba_inventory_summary.inbound_receiving_quantity,
        fba_inventory_summary.inbound_working_quantity,
        fba_inventory_summary.warehouse_damaged_quantity,
        fba_inventory_summary.carrier_damaged_quantity,
        fba_inventory_summary.customer_damaged_quantity,
        fba_inventory_summary.defective_quantity,
        fba_inventory_summary.distributor_damaged_quantity,
        fba_inventory_summary.expired_quantity,
        pivot_researching_quantity.short_term_research_quantity,
        pivot_researching_quantity.mid_term_research_quantity,
        pivot_researching_quantity.long_term_research_quantity
        {% endif %}

    from item
    {% if var('amazon_selling_partner__using_fba_module', true) %}
    left join fba_inventory_summary 
        on fba_inventory_summary.asin = item.asin
        and fba_inventory_summary.source_relation = item.source_relation
    left join pivot_researching_quantity
        on fba_inventory_summary.inventory_summary_id = pivot_researching_quantity.inventory_summary_id
        and fba_inventory_summary.source_relation = pivot_researching_quantity.source_relation
    {% endif %}

)

select *
from joined