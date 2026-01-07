

with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__invoice_line_item_has_tracking_category_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    invoice_id
    
 as 
    
    invoice_id
    
, 
    
    
    line_item_id
    
 as 
    
    line_item_id
    
, 
    
    
    option
    
 as 
    
    option
    
, 
    
    
    tracking_category_id
    
 as 
    
    tracking_category_id
    




        



    
    from base
),

final as (
    
    select 
        invoice_id,
        line_item_id,
        tracking_category_id,
        option as tracking_option_name,
        _fivetran_synced

        


, cast('' as TEXT) as source_relation



        
    from fields
)

select * 
from final