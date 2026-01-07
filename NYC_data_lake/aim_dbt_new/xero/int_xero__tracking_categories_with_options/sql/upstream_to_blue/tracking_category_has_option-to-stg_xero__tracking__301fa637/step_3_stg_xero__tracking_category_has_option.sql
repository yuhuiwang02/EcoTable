

with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__tracking_category_has_option_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    tracking_category_id
    
 as 
    
    tracking_category_id
    
, 
    
    
    tracking_option_id
    
 as 
    
    tracking_option_id
    




        



    
    from base
),

final as (
    
    select
        tracking_category_id, 
        tracking_option_id, 
        _fivetran_synced

        


, cast('' as TEXT) as source_relation



        
    from fields
)

select * 
from final