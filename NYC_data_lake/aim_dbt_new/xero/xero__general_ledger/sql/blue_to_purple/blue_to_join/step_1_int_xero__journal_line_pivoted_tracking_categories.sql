



with journal_line_has_tracking as (

    select *
    from "xero"."public_xero_dev"."stg_xero__journal_line_has_tracking_category"

), tracking_categories_with_options as (

    select *
    from "xero"."public_xero_dev"."int_xero__tracking_categories_with_options"

), journal_tracking as (

    select
        journal_line_has_tracking.journal_id,
        journal_line_has_tracking.journal_line_id,
        journal_line_has_tracking.source_relation,
        tracking_categories_with_options.tracking_category_name,
        tracking_categories_with_options.tracking_option_name
    from journal_line_has_tracking

    left join tracking_categories_with_options
        on journal_line_has_tracking.tracking_category_id = tracking_categories_with_options.tracking_category_id
        and journal_line_has_tracking.tracking_category_option_id = tracking_categories_with_options.tracking_option_id
        and journal_line_has_tracking.source_relation = tracking_categories_with_options.source_relation
),

final as (

    select
        journal_id,
        journal_line_id,
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
      
    
    
  

        
    from journal_tracking
    group by 1,2,3
)

select *
from final