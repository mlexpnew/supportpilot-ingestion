# ReplyPilot draft samples (SP-124)

Generated 2026-11-27 with `llama3.2:3b`, top-3 articles, grouped retrieval. Average latency 3085 ms.

Rating scale: 5 = send as is, 4 = tiny edit, 3 = needs edits, 2 = mostly wrong, 1 = harmful or made up.

| # | Ticket | Citations | Removed | Rating |
|---|---|---|---|---|
| 1 | Hi, my refund was approved last Monday but it stil | kb-refund-timeline | - | 4 |
| 2 | I see two debits of Rs 1,499 on my card for the or | kb-double-charge | - | 4 |
| 3 | My parcel was supposed to arrive 4 days ago and th | kb-late-delivery | - | 3 |
| 4 | I forgot my password. I clicked reset three times  | kb-reset-password | - | 3 |
| 5 | I moved house last week and my order is still show | kb-change-address | - | 3 |
| 6 | I bought the annual plan 3 weeks ago and want to c | - | - | 1 |
| 7 | need a GST invoice with my company name for my las | kb-invoice-download | - | 3 |
| 8 | paisa wapas kab aayega? return pickup ho gaya 5 di | kb-late-delivery, kb-contact-support | - | 2 |
| 9 | I returned 1 of the 3 items from my order and the  | kb-partial-refund | - | 4 |
| 10 | Do you deliver to Dubai? Also can you gift wrap th | - | - | 1 |

**Average rating:** 2.8 / 5 (3 drafts usable with a small edit, 4 need real edits, 1 mostly wrong, 2 wrong or made up)

## Findings

1. **Two drafts give wrong information to customers.** #6 promises no cancellation fee when the policy says one applies, and #10 invents answers about Dubai delivery and gift wrap. Both had **no citations**, so `needs_review` flagged both. Citations work as a safety signal: every uncited draft here was wrong.
2. **Valid citations do not prove correctness.** #8 cites real retrieved articles that do not support its sentences. The citation check only verifies that the id was retrieved, not that the article says what the sentence claims.
3. **Hinglish is still the weak spot.** #8 failed at retrieval, the same weakness the SP-123 eval found.
4. **Drafts often miss the most useful line** (#1, #3, #4, #5, #7). They state a correct fact but skip the action the customer needs (check spam, click Change address, ask support for a corrected invoice).
5. **The model copies the prompt's example.** 7 of 10 drafts start with "Hi there, sorry for the wait", even when nobody waited.
6. **Speed is fine for agent review:** about 2 to 3 seconds per draft after the first (8.8 s cold start) with `llama3.2:3b` on a MacBook.
7. **No made-up citation ids appeared** in this run (Removed = none for all 10), so the removal logic is covered by unit tests only.

## Next steps

1. Add a rule and example for comparisons with numbers (like "after 14 days"), and test #6 again.
2. Make the out-of-scope rule stronger and add an example: "I'm not sure about that, I'll check with the team."
3. Replace "sorry for the wait" in the example with a neutral opening, or use 2 to 3 different examples.
4. Treat a draft with no citations as "do not send" in the agent UI, not just a review flag.
5. Try a larger model (`claude-haiku-5-5` via the Anthropic client, or a 7B to 8B Ollama model) on the same 10 tickets and compare ratings.

## 1. Refund approved but not received

**Ticket:** Hi, my refund was approved last Monday but it still hasn't shown up on my HDFC credit card. How long does it usually take?

**Expected article(s):** kb-refund-timeline

**Retrieved:** kb-refund-timeline, kb-refund-request, kb-partial-refund

**Draft:**

> Hi there, sorry for the wait. According to our refund timeline [kb-refund-timeline], once a refund is approved, it should take 5 to 7 working days for it to appear on your credit card statement. Could you please check with your bank if they have received the refund and when you can expect to see the transaction?

**Citations:** kb-refund-timeline  
**Removed (made up):** none  
**Latency:** 8810 ms, tokens 806 in / 70 out

**Rating (1-5):** 4

**Notes:** Correct and cited: 5 to 7 working days for cards [kb-refund-timeline]. Misses the most useful part for this customer: after 10 working days support can share a refund reference number for the bank to trace.

## 2. Charged twice

**Ticket:** I see two debits of Rs 1,499 on my card for the order I placed yesterday. I only ordered once. Please fix this.

**Expected article(s):** kb-double-charge

**Retrieved:** kb-double-charge, kb-contact-support, kb-update-payment

**Draft:**

> Hi there, sorry to hear that you've been charged twice for your recent order. According to our policy, seeing two charges usually means one is a temporary authorisation hold, not a real payment [kb-double-charge]. Could you please wait for 3 working days and check My Orders to confirm you placed only one order? If both charges are still there after that time, we'll be happy to refund the duplicate amount.

**Citations:** kb-double-charge  
**Removed (made up):** none  
**Latency:** 2722 ms, tokens 789 in / 87 out

**Rating (1-5):** 4

**Notes:** Correct: temporary hold, wait 3 working days, check My Orders [kb-double-charge]. Should ask for the order number and the date of each charge if both stay, as the article says. Last two sentences have no citation.

## 3. Parcel late, tracking stuck

**Ticket:** My parcel was supposed to arrive 4 days ago and the tracking hasn't changed since Friday. What's going on?

**Expected article(s):** kb-late-delivery, kb-track-order

**Retrieved:** kb-track-order, kb-late-delivery, kb-change-address

**Draft:**

> Hi there, sorry for the wait. If your parcel is more than 5 working days late, we will open a courier investigation and, if it is lost, send a replacement or refund [kb-late-delivery]. Could you please share your order number so I can check the status for you?

**Citations:** kb-late-delivery  
**Removed (made up):** none  
**Latency:** 2228 ms, tokens 807 in / 62 out

**Rating (1-5):** 3

**Notes:** The 5-working-day investigation rule is correct [kb-late-delivery], but the parcel is only 4 days late. Misses the better fit from kb-track-order (retrieved first): tracking unchanged for 3 working days means support checks with the courier. Asking for the order number is right.

## 4. Password reset email missing

**Ticket:** I forgot my password. I clicked reset three times but the email never comes. Can you help?

**Expected article(s):** kb-reset-password

**Retrieved:** kb-reset-password, kb-update-email, kb-cancel-subscription

**Draft:**

> Hi there, sorry for the wait. It seems like the reset link didn't arrive in your inbox. Could you please check if you're using a different email address associated with your account? If that's not it, we can try to verify your identity and help you recover your password [kb-reset-password].

**Citations:** kb-reset-password  
**Removed (made up):** none  
**Latency:** 2225 ms, tokens 783 in / 63 out

**Rating (1-5):** 3

**Notes:** Correct that the account may use a different email and support can verify identity [kb-reset-password]. Misses the most common fix: check the spam folder, and the link expires in 30 minutes. "Sorry for the wait" makes no sense here.

## 5. Change address before shipping

**Ticket:** I moved house last week and my order is still showing Processing. Can I send it to my new address?

**Expected article(s):** kb-change-address

**Retrieved:** kb-change-address, kb-track-order, kb-late-delivery

**Draft:**

> Hi there, sorry for the wait. You can change your delivery address only before the order is dispatched, so if you haven't received a shipping confirmation email yet, it's still possible to update your address [kb-change-address]. If you're concerned about the status of your order, I'd be happy to check on it with you - could you please share your order number?

**Citations:** kb-change-address  
**Removed (made up):** none  
**Latency:** 2491 ms, tokens 805 in / 77 out

**Rating (1-5):** 3

**Notes:** Correct that the address can change only before dispatch [kb-change-address]. The status is Processing, so the real answer is: My Orders, open the order, click Change address. The draft skips these steps, adds an unsupported shipping-email rule and asks for an order number it does not need.

## 6. Cancel annual plan after 3 weeks

**Ticket:** I bought the annual plan 3 weeks ago and want to cancel now. Will I be charged anything for cancelling?

**Expected article(s):** kb-cancel-fees, kb-cancel-subscription

**Retrieved:** kb-cancel-fees, kb-cancel-subscription, kb-cancel-order

**Draft:**

> Hi there, sorry for the wait. According to our policies, annual plans cancelled after 14 days are refunded minus one month's fee. Since you've had your plan for only 3 weeks, no cancellation fee will be applied when you cancel now. Could you please confirm if this is what you were expecting?

**Citations:** none  
**Removed (made up):** none  
**Latency:** 2226 ms, tokens 790 in / 64 out

**Rating (1-5):** 1

**Notes:** WRONG and harmful. kb-cancel-fees says annual plans cancelled after 14 days are refunded minus one month's fee. 3 weeks is 21 days, so the fee DOES apply, but the draft promises no fee. No citations, so it was flagged needs_review.

## 7. GST invoice for company

**Ticket:** need a GST invoice with my company name for my last order, how do i get it

**Expected article(s):** kb-invoice-download

**Retrieved:** kb-invoice-download, kb-contact-support, kb-delete-account

**Draft:**

> Hi there, sorry for the wait. To obtain a GST invoice with your company name for your last order, please go to My Orders and open the order. Click Download invoice to get a PDF. For a business invoice, add your company name and tax ID under Account, then Billing details before ordering [kb-invoice-download]. Could you also confirm that you have an active account with us so I can assist you further?

**Citations:** kb-invoice-download  
**Removed (made up):** none  
**Latency:** 2592 ms, tokens 757 in / 86 out

**Rating (1-5):** 3

**Notes:** Steps are correct and cited [kb-invoice-download]. Misses the key point: company details must be added before ordering, and issued invoices cannot be edited, so for a past order support must issue a corrected invoice. Asking if they have an active account is unnecessary.

## 8. Hinglish refund question

**Ticket:** paisa wapas kab aayega? return pickup ho gaya 5 din pehle, abhi tak kuch nahi aaya

**Expected article(s):** kb-refund-timeline, kb-refund-request

**Retrieved:** kb-late-delivery, kb-business-hours, kb-contact-support

**Draft:**

> Hi there, sorry for the wait. Since your return pickup was scheduled 5 days ago and you haven't received it yet, we'll need to investigate further. Could you please share your order number so I can check on its status? We'll look into this as soon as possible [kb-late-delivery] [kb-contact-support].

**Citations:** kb-late-delivery, kb-contact-support  
**Removed (made up):** none  
**Latency:** 2364 ms, tokens 791 in / 70 out

**Rating (1-5):** 2

**Notes:** Retrieval missed (Hinglish, the known weak spot): kb-refund-timeline was not retrieved. The model misread the ticket as a missing pickup, not a missing refund. The citations are valid ids but do not support the sentences, which our check cannot catch.

## 9. Partial refund looks too small

**Ticket:** I returned 1 of the 3 items from my order and the refund is much less than what I paid for it. Why?

**Expected article(s):** kb-partial-refund

**Retrieved:** kb-partial-refund, kb-refund-request, kb-double-charge

**Draft:**

> Hi there, sorry to hear that your partial refund doesn't match what you paid for the item. According to our policy, a partial refund returns part of the order amount instead of the full price, and it's calculated by refunding the price you paid for the returned item, minus any discount shared across the order [kb-partial-refund]. Could you please check your refund email to see the breakdown of how this was calculated? If the amount still looks wrong, we'd be happy to review it with you.

**Citations:** kb-partial-refund  
**Removed (made up):** none  
**Latency:** 3041 ms, tokens 807 in / 105 out

**Rating (1-5):** 4

**Notes:** Correct and cited: refund for the returned item minus shared discount, breakdown in the refund email [kb-partial-refund]. Tiny edit: tell them to reply to the refund email to get it reviewed, as the article says.

## 10. Out of scope: international + gift wrap

**Ticket:** Do you deliver to Dubai? Also can you gift wrap the order and add a note?

**Expected article(s):** -

**Retrieved:** kb-change-address, kb-cancel-order, kb-late-delivery

**Draft:**

> Hi there,
> We do not currently deliver to Dubai. However, we would be happy to help you with your order. Unfortunately, gift wrapping and adding a note are services that require an additional charge. Could you please let me know if you'd like me to look into this further?

**Citations:** none  
**Removed (made up):** none  
**Latency:** 2157 ms, tokens 798 in / 59 out

**Rating (1-5):** 1

**Notes:** MADE UP. No article covers international delivery or gift wrap, yet the draft says we do not deliver to Dubai and gift wrap costs extra. It should say it will check with the team. No citations, so it was flagged needs_review.