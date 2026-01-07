
-- This model is only necessary when unioning multiple sources and will therefore be disabled when that is not the case






    select
            "order_item_id",
  "amazon_order_id",
  "is_gift",
  "is_transparency",
  "asin",
  "buyer_requested_cancel_is_buyer_requested_cancel",
  "buyer_requested_cancel_buyer_cancel_reason",
  "condition_id",
  "condition_note",
  "condition_subtype_id",
  "deemed_reseller_category",
  "ioss_number",
  "item_approval_context_approval_status",
  "item_approval_context_approval_type",
  "item_price_amount",
  "item_price_currency_code",
  "item_tax_amount",
  "item_tax_currencycode",
  "points_granted_monetary_amount",
  "points_granted_monetary_currency_code",
  "points_granted_points_number",
  "price_designation",
  "product_info_detail_number_of_items",
  "promotion_discount_amount",
  "promotion_discount_currency_code",
  "promotion_discount_tax_currency_code",
  "promotion_discount_tax_amount",
  "quantity_ordered",
  "quantity_shipped",
  "scheduled_delivery_end_date",
  "scheduled_delivery_start_date",
  "seller_sku",
  "serial_number_required",
  "shipping_discount_amount",
  "shipping_discount_currency_code",
  "shipping_discount_tax_amount",
  "shipping_discount_tax_currency_code",
  "shipping_price_amount",
  "shipping_price_currency_code",
  "shipping_tax_amount",
  "shipping_tax_currency_code",
  "store_chain_store_id",
  "tax_collection_model",
  "tax_collection_responsible_party",
  "title"
        from "amazon_selling_partner"."public"."order_item" as source_table
    
    