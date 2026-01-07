
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "inventory_summary_id",
  "name",
  "quantity"
        from "amazon_selling_partner"."public"."fba_inventory_researching_quantity_entry" as source_table
    
    