import { z } from 'zod'

export const relatedQuestionsSchema = z.object({
  items: z.array(
    z.object({
      query: z.string().min(1)
    })
  ).length(3) // This enforces exactly 3 items
})

export type RelatedQuestions = z.infer<typeof relatedQuestionsSchema> 