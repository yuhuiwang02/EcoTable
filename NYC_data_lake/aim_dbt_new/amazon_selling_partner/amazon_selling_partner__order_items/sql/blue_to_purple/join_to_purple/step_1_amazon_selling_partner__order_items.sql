

with  __dbt__cte__int_amazon_selling_partner__item as (


with item_summary as (
    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_summary"
),

item_product_type as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_product_type" 
),

item_image as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_image"
),

item_images as (

    select 
        source_relation,
        asin,
        marketplace_id,
        count(*) as count_images,
        sum(case when variant = 'SWATCH' then 1 else 0 end) as count_swatch_images

    from item_image 
    group by 1,2,3
),

item_display_group_sales_rank as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_display_group_sales_rank" 
),

item_classification_sales_rank as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_classification_sales_rank" 
),

item_relationship as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_relationship" 
),

parent_variation_relationship as (

    select *
    from item_relationship
    where type = 'VARIATION'
),

package_hierarchy_relationship as (

    select *
    from item_relationship
    where type = 'PACKAGE_HIERARCHY'
),

item_dimension as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_dimension" 
),

item_identifier as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_identifier" 
),

item_identifiers as (

    select 
        asin,
        source_relation,
        marketplace_id
        
        
            , cast(max(case when identifier_type = 'SKU' then identifier end) as TEXT) as sku
        
            , cast(max(case when identifier_type = 'EAN' then identifier end) as TEXT) as ean
        
            , cast(max(case when identifier_type = 'GTIN' then identifier end) as TEXT) as gtin
        
            , cast(max(case when identifier_type = 'ISBN' then identifier end) as TEXT) as isbn
        
            , cast(max(case when identifier_type = 'JAN' then identifier end) as TEXT) as jan
        
            , cast(max(case when identifier_type = 'MINSAN' then identifier end) as TEXT) as minsan
        
            , cast(max(case when identifier_type = 'UPC' then identifier end) as TEXT) as upc
        
    from item_identifier
    group by 1,2,3
),

joined as (

    select 
        item_summary.source_relation,
        item_summary.marketplace_id,
        item_summary.asin,
        item_summary.item_name,
        item_summary.display_name,
        item_summary.brand,
        item_summary.color,
        item_summary.size,
        item_summary.style,
        item_summary.package_quantity,
        item_summary.manufacturer,
        item_summary.contributors,
        item_product_type.product_type,
        item_summary.item_classification,
        item_summary.classification_id,
        item_classification_sales_rank.link as classification_sales_rank_link,
        item_classification_sales_rank.rank as classification_sales_rank,
        item_summary.website_display_group,
        item_summary.website_display_group_name,
        item_display_group_sales_rank.link as website_display_group_sales_rank_link,
        item_display_group_sales_rank.rank as website_display_group_sales_rank,
        item_summary.release_date,
        item_summary.is_memorabilia,
        item_summary.is_adult_product,
        item_summary.is_autographed,
        item_summary.is_trade_in_eligible,

        item_summary.model_number,
        item_summary.part_number,
        parent_variation_relationship.parent_asin as parent_variation_asin,
        package_hierarchy_relationship.parent_asin as parent_package_container_asin,
        item_identifiers.sku,
        item_identifiers.ean,
        item_identifiers.gtin, 
        item_identifiers.isbn, 
        item_identifiers.jan,
        item_identifiers.minsan, 
        item_identifiers.upc,
        
        item_images.count_images,
        item_images.count_swatch_images,
        item_dimension.item_height_unit,
        item_dimension.item_height_value,
        item_dimension.item_length_unit,
        item_dimension.item_length_value,
        item_dimension.item_weight_unit,
        item_dimension.item_weight_value,
        item_dimension.item_width_unit,
        item_dimension.item_width_value,
        item_dimension.package_height_unit,
        item_dimension.package_height_value,
        item_dimension.package_length_unit,
        item_dimension.package_length_value,
        item_dimension.package_weight_unit,
        item_dimension.package_weight_value,
        item_dimension.package_width_unit,
        item_dimension.package_width_value

    from item_summary
    left join item_product_type
        on item_summary.asin = item_product_type.asin 
        and item_summary.marketplace_id = item_product_type.marketplace_id
        and item_summary.source_relation = item_product_type.source_relation
    left join item_images
        on item_summary.asin = item_images.asin 
        and item_summary.marketplace_id = item_images.marketplace_id
        and item_summary.source_relation = item_images.source_relation
    left join item_display_group_sales_rank
        on item_summary.asin = item_display_group_sales_rank.asin 
        and item_summary.website_display_group = item_display_group_sales_rank.website_display_group
        and item_summary.source_relation = item_display_group_sales_rank.source_relation
    left join item_classification_sales_rank
        on item_summary.asin = item_classification_sales_rank.asin 
        and item_summary.classification_id = item_classification_sales_rank.classification_id
        and item_summary.source_relation = item_classification_sales_rank.source_relation
    left join parent_variation_relationship
        on item_summary.asin = parent_variation_relationship.child_asin
        and item_summary.source_relation = parent_variation_relationship.source_relation
    left join package_hierarchy_relationship
        on item_summary.asin = package_hierarchy_relationship.child_asin
        and item_summary.source_relation = package_hierarchy_relationship.source_relation
    left join item_identifiers
        on item_summary.asin = item_identifiers.asin 
        and item_summary.marketplace_id = item_identifiers.marketplace_id
        and item_summary.source_relation = item_identifiers.source_relation
    left join item_dimension 
        on item_summary.asin = item_dimension.asin 
        and item_summary.marketplace_id = item_dimension.marketplace_id
        and item_summary.source_relation = item_dimension.source_relation
)

select *
from joined
), orders as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__orders"
),

order_item as (

    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__order_item"
),


item as (

    select *
    from __dbt__cte__int_amazon_selling_partner__item
),


order_item_promotion_id as (
    
    select *
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__order_item_promotion_id"
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

        
        , item.item_name,
        item.display_name,
        item.brand,
        item.color,
        item.size,
        item.style,
        item.product_type,
        item.package_quantity,
        item.item_classification
        

    from order_item 
    
    left join item 
        on order_item.asin = item.asin 
        and order_item.source_relation = item.source_relation
    
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