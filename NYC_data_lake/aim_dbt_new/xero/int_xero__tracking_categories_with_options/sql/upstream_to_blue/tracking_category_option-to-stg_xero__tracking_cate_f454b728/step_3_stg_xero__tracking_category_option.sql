

with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__tracking_category_option_tmp"

),

fields as (

    select
        
    
    
    _fivetran_synced
    
 as 
    
    _fivetran_synced
    
, 
    
    
    has_validation_errors
    
 as 
    
    has_validation_errors
    
, 
    
    
    is_active
    
 as 
    
    is_active
    
, 
    
    
    is_archived
    
 as 
    
    is_archived
    
, 
    
    
    is_deleted
    
 as 
    
    is_deleted
    
, 
    
    
    name
    
 as 
    
    name
    
, 
    
    
    status
    
 as 
    
    status
    
, 
    
    
    tracking_option_id
    
 as 
    
    tracking_option_id
    




        



    
    from base
),

final as (
    
    select
        tracking_option_id,
        name as tracking_option_name,
        status,
        has_validation_errors,
        is_active,
        is_archived,
        is_deleted,
        _fivetran_synced

        


, cast('' as TEXT) as source_relation



        
    from fields
)

select * 
from final