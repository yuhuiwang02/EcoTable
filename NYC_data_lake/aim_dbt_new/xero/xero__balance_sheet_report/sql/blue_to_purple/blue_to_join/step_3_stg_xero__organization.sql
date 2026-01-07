with base as (

    select * 
    from "xero"."public_xero_dev"."stg_xero__organization_tmp"

),

fields as (

    select
        
    cast(null as timestamp) as 
    
    _fivetran_synced
    
 , 
    cast(null as TEXT) as 
    
    apikey
    
 , 
    cast(null as TEXT) as 
    
    base_currency
    
 , 
    cast(null as TEXT) as 
    
    class
    
 , 
    cast(null as TEXT) as 
    
    country_code
    
 , 
    cast(null as timestamp) as 
    
    created_date_utc
    
 , 
    cast(null as TEXT) as 
    
    default_purchases_tax
    
 , 
    cast(null as TEXT) as 
    
    default_sales_tax
    
 , 
    cast(null as TEXT) as 
    
    edition
    
 , 
    cast(null as TEXT) as 
    
    employer_identification_number
    
 , 
    cast(null as date) as 
    
    end_of_year_lock_date
    
 , 
    
    
    financial_year_end_day
    
 as 
    
    financial_year_end_day
    
, 
    
    
    financial_year_end_month
    
 as 
    
    financial_year_end_month
    
, 
    cast(null as boolean) as 
    
    is_demo_company
    
 , 
    cast(null as TEXT) as 
    
    legal_name
    
 , 
    cast(null as TEXT) as 
    
    line_of_business
    
 , 
    cast(null as TEXT) as 
    
    name
    
 , 
    cast(null as TEXT) as 
    
    organisation_entity_type
    
 , 
    
    
    organisation_id
    
 as 
    
    organisation_id
    
, 
    cast(null as TEXT) as 
    
    organisation_status
    
 , 
    cast(null as TEXT) as 
    
    organisation_type
    
 , 
    cast(null as boolean) as 
    
    pays_tax
    
 , 
    cast(null as date) as 
    
    period_lock_date
    
 , 
    cast(null as TEXT) as 
    
    registration_number
    
 , 
    cast(null as TEXT) as 
    
    sales_tax_basis
    
 , 
    cast(null as TEXT) as 
    
    sales_tax_period
    
 , 
    cast(null as TEXT) as 
    
    short_code
    
 , 
    cast(null as TEXT) as 
    
    tax_number
    
 , 
    cast(null as TEXT) as 
    
    tax_number_name
    
 , 
    cast(null as TEXT) as 
    
    timezone
    
 , 
    cast(null as TEXT) as 
    
    version
    
 



        




    from base
),

final as (
    
    select 
        organisation_id,
        financial_year_end_month,
        financial_year_end_day

        


, cast('' as TEXT) as source_relation



        
    from fields
)

select * from final