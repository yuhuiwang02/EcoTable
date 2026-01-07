



with invoice_line_item_has_tracking as (

    select *
    from "xero"."public_xero_dev"."stg_xero__invoice_line_item_has_tracking_category"

), tracking_categories_with_options as (

    select *
    from "xero"."public_xero_dev"."int_xero__tracking_categories_with_options"

), invoice_tracking as (

    select
        invoice_line_item_has_tracking.invoice_id,
        invoice_line_item_has_tracking.line_item_id,
        invoice_line_item_has_tracking.source_relation,
        tracking_categories_with_options.tracking_category_name,
        tracking_categories_with_options.tracking_option_name
    from invoice_line_item_has_tracking

    left join tracking_categories_with_options
        on invoice_line_item_has_tracking.tracking_category_id = tracking_categories_with_options.tracking_category_id
        and invoice_line_item_has_tracking.tracking_option_name = tracking_categories_with_options.tracking_option_name
        and invoice_line_item_has_tracking.source_relation = tracking_categories_with_options.source_relation

), final as (

    select
        invoice_id,
        line_item_id,
        source_relation
        
        ,   
  
    max(
      
      case
      when tracking_category_name = 'Region'
        then tracking_option_name
      else null
      end
    )
    
      
        as region
      
    
    ,
  
    max(
      
      case
      when tracking_category_name = 'Department'
        then tracking_option_name
      else null
      end
    )
    
      
        as department
      
    
    ,
  
    max(
      
      case
      when tracking_category_name = 'Location'
        then tracking_option_name
      else null
      end
    )
    
      
        as location
      
    
    ,
  
    max(
      
      case
      when tracking_category_name = 'Project'
        then tracking_option_name
      else null
      end
    )
    
      
        as project
      
    
    
  

        
    from invoice_tracking
    group by 1,2,3
)

select *
from final