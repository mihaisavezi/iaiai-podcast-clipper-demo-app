"use server";

import { hashPassword } from "~/lib/auth";
import { signupSchema, type SignupFormValues } from "~/schemas/auth";
import { db } from "~/server/db";
import Stripe from "stripe";


type SignupResult = {
  success: boolean;
  error?: string;
};

export async function signUp(data: SignupFormValues): Promise<SignupResult> {
  const validationResults = signupSchema.safeParse(data);

  if (!validationResults.success) {
    return {
      success: false,
      error: validationResults.error.issues[0].message || "Invalid Input",
    };
  }

  const { email, password } = validationResults.data;

  try {
    const existingUser = await db.user.findUnique({
      where: { email },
    });

    if(existingUser) {
        return {
            success: false,
            error: "User already exists"
        }
    }

    const hashedPassword=  await hashPassword(password);

    // const stripe = new Stripe("TODO: stripe key");

    // const stripeCustomer = await stripe.customers.create({
    //     email: email.toLowerCase()
    // })

    await db.user.create({
        data: {
            email: email.toLowerCase(),
            password: hashedPassword,
            // stripeCustomerId: stripeCustomer.id)
        }})

    return { success: true };



 } catch (error) {
    return { success: false, error: "Something went wrong during signup" };
  }
}
