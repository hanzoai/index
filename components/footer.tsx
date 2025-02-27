'use client'
import { Check } from 'lucide-react'
import Link from 'next/link'
import React, { useEffect, useState } from 'react'
import { SiDiscord, SiGithub, SiX } from 'react-icons/si'
import { Button } from './ui/button'
import { IconLogo } from './ui/icons'
const Footer: React.FC = () => {
  const [showFooter, setShowFooter] = useState(false)

  useEffect(() => {
    const handleMouseWheel = (event: WheelEvent) => {
      if (event.deltaY > 0) {
        setShowFooter(true)
      } else if (event.deltaY < 0 && window.pageYOffset === 0) {
        setShowFooter(false)
      }
    }

    window.addEventListener('wheel', handleMouseWheel)

    return () => {
      window.removeEventListener('wheel', handleMouseWheel)
    }
  }, [])

  return (
    <footer className={`fixed bottom-0 left-0 right-0 transition-transform duration-300 bg-background ${showFooter ? 'translate-y-0' : 'translate-y-full'}`}>
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex justify-between items-center">
          <div className="col-span-2 md:col-span-4 lg:col-span-1">
            <div className="flex items-center space-x-2">
              <IconLogo className="h-4 w-4" />
              <span className="text-foreground font-bold text-xl">Hanzo</span>
            </div>
            <a
              href="https://hanzo.industries/status"
              target="_blank"
              className="inline-flex items-center space-x-2 text-gray-400 hover:text-foreground mt-1"
            >
              <Check className="h-4 w-4 text-green-500" />
              <span>All systems operational</span>
            </a>
          </div>
          <div className="flex space-x-4">
            <Button
              variant={'ghost'}
              size={'icon'}
              className="text-muted-foreground/50"
            >
              <Link href="#" target="_blank">
                <SiDiscord size={18} />
              </Link>
            </Button>
            <Button
              variant={'ghost'}
              size={'icon'}
              className="text-muted-foreground/50"
            >
              <Link href="https://x.com/hanzoai" target="_blank">
                <SiX size={18} />
              </Link>
            </Button>
            <Button
              variant={'ghost'}
              size={'icon'}
              className="text-muted-foreground/50"
            >
              <Link href="https://github.com/hanzoai" target="_blank">
                <SiGithub size={18} />
              </Link>
            </Button>
          </div>
        </div>
        <div className="mt-4 border-t border-gray-800 pt-4">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="text-muted-foreground text-sm">
              © {new Date().getFullYear()} Hanzo Industries, Inc. All rights reserved.
            </div>
            <div className="flex space-x-6">
              <a
                href="https://hanzo.industries/privacy"
                target="_blank"
                className="text-muted-foreground hover:text-foreground text-sm"
              >
                Privacy Policy
              </a>
              <a
                href="https://hanzo.industries/terms"
                target="_blank"
                className="text-muted-foreground hover:text-foreground text-sm"
              >
                Terms of Service
              </a>
              <a
                href="https://hanzo.industries/security"
                target="_blank"
                className="text-muted-foreground hover:text-foreground text-sm"
              >
                Security
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
