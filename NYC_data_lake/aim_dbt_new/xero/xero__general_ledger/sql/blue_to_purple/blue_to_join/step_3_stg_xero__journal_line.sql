with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__journal_line_tmp"

),

fields as (

    select
        
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    
    
    account_code
    
 as 
    
    account_code
    
, 
    
    
    account_id
    
 as 
    
    account_id
    
, 
    
    
    account_name
    
 as 
    
    account_name
    
, 
    
    
    account_type
    
 as 
    
    account_type
    
, 
    
    
    description
    
 as 
    
    description
    
, 
    
    
    gross_amount
    
 as 
    
    gross_amount
    
, 
    
    
    journal_id
    
 as 
    
    journal_id
    
, 
    
    
    journal_line_id
    
 as 
    
    journal_line_id
    
, 
    
    
    net_amount
    
 as 
    
    net_amount
    
, 
    
    
    tax_amount
    
 as 
    
    tax_amount
    
, 
    
    
    tax_name
    
 as 
    
    tax_name
    
, 
    
    
    tax_type
    
 as 
    
    tax_type
    




        




    from base
),

final as (
    
    select 
        journal_line_id,
        account_code,
        account_id,
        account_name,
        account_type,
        description,
        gross_amount,
        journal_id,
        net_amount,
        tax_amount,
        tax_name,
        tax_type

        


, cast('' as TEXT) as source_relation



        
    from fields
)

select * from final