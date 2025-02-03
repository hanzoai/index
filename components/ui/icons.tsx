'use client'

import { cn } from '@/lib/utils'

function IconLogo({ className, ...props }: React.ComponentProps<'svg'>) {
  return (
    <svg
      fill="currentColor"
      viewBox="0 0 256 256"
      role="img"
      xmlns="http://www.w3.org/2000/svg"
      className={cn('h-4 w-4', className)}
      {...props}
    >
      <g transform="translate(0.000000,128.000000) scale(0.100000,-0.100000)"
        fill="#000000" stroke="none">
        <path d="M0 640 l0 -640 640 0 640 0 0 640 0 640 -640 0 -640 0 0 -640z m518
248 l3 -118 119 0 120 0 0 120 0 120 125 0 125 0 0 -122 0 -123 -122 -122
-123 -123 123 0 122 0 0 -125 0 -125 -125 0 -125 0 0 125 0 126 -121 -3 -122
-3 -1 -120 -1 -120 -122 -3 -123 -3 0 129 0 128 120 119 c66 65 120 120 120
122 0 1 -54 3 -120 3 l-120 0 0 120 0 121 123 -3 122 -3 3 -117z"/>
      </g>
    </svg>
  )
}

export { IconLogo }
