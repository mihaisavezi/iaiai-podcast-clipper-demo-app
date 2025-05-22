import Link from "next/link";

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c] text-white">
      <div className="container flex flex-col items-center justify-center gap-8 px-4 py-16">
      <h1 className="text-5xl font-extrabold tracking-tight text-white sm:text-[4rem] text-center">
      Podcast <span className="text-[#9750dd]">Clipper</span>
      </h1>
      <p className="text-xl text-center max-w-2xl">
      Transform your favorite podcast moments into shareable clips with just a few clicks
      </p>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:gap-8 max-w-4xl">
      <div className="flex flex-col gap-4 rounded-xl bg-white/10 p-6 text-white hover:bg-white/20">
      <h3 className="text-2xl font-bold">Easy Clipping</h3>
      <div className="text-lg">
        Select your favorite podcast moments and create perfect clips in seconds
      </div>
      </div>
      <div className="flex flex-col gap-4 rounded-xl bg-white/10 p-6 text-white hover:bg-white/20">
      <h3 className="text-2xl font-bold">Share Anywhere</h3>
      <div className="text-lg"></div>
        Share your clips instantly on social media or with your friends
      </div>
      </div>
      </div>
      <Link
      href="/dashboard"
      className="mt-8 rounded-full bg-[#9750dd] px-8 py-3 text-lg font-semibold text-white hover:bg-[#7e3ab7] transition-colors"
      >
      Get Started
      </Link>
    </main>
  );
}
