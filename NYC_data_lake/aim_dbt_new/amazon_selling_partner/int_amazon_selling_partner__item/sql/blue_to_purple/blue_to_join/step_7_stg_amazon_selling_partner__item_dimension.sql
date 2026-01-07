

with base as (

    select * 
    from "amazon_selling_partner"."public_amazon_selling_partner_dev"."stg_amazon_selling_partner__item_dimension_base"
),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    asin
    
 as 
    
    asin
    
, 
    
    
    item_height_unit
    
 as 
    
    item_height_unit
    
, 
    
    
    item_height_value
    
 as 
    
    item_height_value
    
, 
    
    
    item_length_unit
    
 as 
    
    item_length_unit
    
, 
    
    
    item_length_value
    
 as 
    
    item_length_value
    
, 
    
    
    item_weight_unit
    
 as 
    
    item_weight_unit
    
, 
    
    
    item_weight_value
    
 as 
    
    item_weight_value
    
, 
    
    
    item_width_unit
    
 as 
    
    item_width_unit
    
, 
    
    
    item_width_value
    
 as 
    
    item_width_value
    
, 
    
    
    marketplace_id
    
 as 
    
    marketplace_id
    
, 
    
    
    package_height_unit
    
 as 
    
    package_height_unit
    
, 
    
    
    package_height_value
    
 as 
    
    package_height_value
    
, 
    
    
    package_length_unit
    
 as 
    
    package_length_unit
    
, 
    
    
    package_length_value
    
 as 
    
    package_length_value
    
, 
    
    
    package_weight_unit
    
 as 
    
    package_weight_unit
    
, 
    
    
    package_weight_value
    
 as 
    
    package_weight_value
    
, 
    
    
    package_width_unit
    
 as 
    
    package_width_unit
    
, 
    
    
    package_width_value
    
 as 
    
    package_width_value
    



        
        
, 'amazon_selling_partner' || '.'|| 'public' as source_relation

        
    from base
),

final as (
    
    select 
        source_relation, 
        _fivetran_synced,
        cast(asin as TEXT) as asin,
        marketplace_id,
        item_height_unit,
        item_height_value,
        item_length_unit,
        item_length_value,
        item_weight_unit,
        item_weight_value,
        item_width_unit,
        item_width_value,
        package_height_unit,
        package_height_value,
        package_length_unit,
        package_length_value,
        package_weight_unit,
        package_weight_value,
        package_width_unit,
        package_width_value
    from fields
)

select *
from final