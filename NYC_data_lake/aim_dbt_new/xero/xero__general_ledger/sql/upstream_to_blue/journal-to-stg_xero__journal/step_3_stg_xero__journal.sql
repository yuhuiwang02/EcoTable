with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__journal_tmp"

),

fields as (

    select
        
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    
    
    created_date_utc
    
 as 
    
    created_date_utc
    
, 
    
    
    journal_date
    
 as 
    
    journal_date
    
, 
    
    
    journal_id
    
 as 
    
    journal_id
    
, 
    
    
    journal_number
    
 as 
    
    journal_number
    
, 
    
    
    reference
    
 as 
    
    reference
    
, 
    
    
    source_id
    
 as 
    
    source_id
    
, 
    
    
    source_type
    
 as 
    
    source_type
    




        




    from base
),

final as (
    
    select 
        journal_id,
        created_date_utc,
        journal_date,
        journal_number,
        reference,
        source_id,
        source_type

        


, cast('' as TEXT) as source_relation



        
    from fields
)

select * from final