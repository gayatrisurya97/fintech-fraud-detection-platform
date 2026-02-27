-- models/staging/stg_transactions.sql
-- PURPOSE: Clean the raw data
-- Fix column names, data types, remove invalid records

with source as (

    select * from public.raw_transactions

),

cleaned as (

    select
        -- Transaction identifiers
        step                                        as transaction_hour,
        type                                        as transaction_type,
        
        -- Amount
        round(cast(amount as numeric), 2)           as amount,
        
        -- Sender details
        nameorig                                    as sender_id,
        round(cast(oldbalanceorg as numeric), 2)    as sender_balance_before,
        round(cast(newbalanceorig as numeric), 2)   as sender_balance_after,
        
        -- Receiver details
        namedest                                    as receiver_id,
        round(cast(oldbalancedest as numeric), 2)   as receiver_balance_before,
        round(cast(newbalancedest as numeric), 2)   as receiver_balance_after,
        
        -- Fraud labels
        isfraud                                     as is_fraud,
        isflaggedfraud                              as is_flagged_fraud

    from source

    where amount > 0
      and amount is not null

)

select * from cleaned